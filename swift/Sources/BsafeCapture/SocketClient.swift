import Foundation

class SocketClient {
    private let path: String
    private var fd: Int32 = -1
    // IMPORTANT: read and write use separate locks because the shutdown listener
    // blocks on readMessage() waiting for commands from Python. A single shared
    // lock would deadlock the frame callback's sendMessage() calls.
    private let readLock = NSLock()
    private let writeLock = NSLock()

    init(path: String) {
        self.path = path
    }

    func connect() throws {
        fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw NSError(
                domain: "SocketClient", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create socket"]
            )
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = path.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            throw NSError(
                domain: "SocketClient", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Socket path too long"]
            )
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                for (i, byte) in pathBytes.enumerated() {
                    dest[i] = byte
                }
            }
        }

        let addrLen = socklen_t(
            MemoryLayout<sa_family_t>.size + pathBytes.count)
        let result = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                Darwin.connect(fd, sockaddrPtr, addrLen)
            }
        }

        guard result == 0 else {
            let errMsg = String(cString: strerror(errno))
            close(fd)
            fd = -1
            throw NSError(
                domain: "SocketClient", code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey: "Failed to connect: \(errMsg)"
                ]
            )
        }

        // Large send buffer for multi-MB raw frames; ignore failure.
        var sndbuf: Int32 = 8 * 1024 * 1024
        _ = setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, socklen_t(MemoryLayout<Int32>.size))
    }

    func sendMessage(type: UInt8, payload: Data) throws {
        writeLock.lock()
        defer { writeLock.unlock() }

        let length = UInt32(1 + payload.count).bigEndian
        var header = Data()
        header.append(contentsOf: withUnsafeBytes(of: length) { Array($0) })
        header.append(type)

        try send(data: header)
        if !payload.isEmpty {
            try send(data: payload)
        }
    }

    /// Send MSG_FRAME_RAW (0x02) without concatenating the pixels into a new
    /// buffer: framing header + 24-byte `!IIIqI` header, then the pixels.
    func sendFrameRaw(displayID: UInt32, width: UInt32, height: UInt32, ptsNs: Int64, seq: UInt32, pixels: Data) throws {
        writeLock.lock()
        defer { writeLock.unlock() }

        let length = UInt32(1 + 24 + pixels.count)
        var header = Data()
        header.reserveCapacity(5 + 24)
        SocketClient.appendBE(&header, length)
        header.append(0x02)
        SocketClient.appendBE(&header, displayID)
        SocketClient.appendBE(&header, width)
        SocketClient.appendBE(&header, height)
        SocketClient.appendBE(&header, ptsNs)
        SocketClient.appendBE(&header, seq)

        try send(data: header)
        if !pixels.isEmpty {
            try send(data: pixels)
        }
    }

    private static func appendBE<T: FixedWidthInteger>(_ data: inout Data, _ value: T) {
        var be = value.bigEndian
        withUnsafeBytes(of: &be) { data.append(contentsOf: $0) }
    }

    /// Max message payload size (256 MiB — must match Python's MAX_MESSAGE_SIZE)
    private static let maxMessageSize = 256 * 1024 * 1024

    func readMessage() throws -> (type: UInt8, payload: Data) {
        readLock.lock()
        defer { readLock.unlock() }

        let header = try recv(exactly: 5)
        let length = header.withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let msgType = header[4]
        let payloadSize = Int(length) - 1
        guard payloadSize >= 0 else {
            throw NSError(
                domain: "SocketClient", code: 7,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Invalid message length: \(length)"
                ]
            )
        }
        guard payloadSize <= SocketClient.maxMessageSize else {
            throw NSError(
                domain: "SocketClient", code: 6,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Message too large: \(payloadSize) bytes exceeds \(SocketClient.maxMessageSize) byte limit"
                ]
            )
        }
        let payload = payloadSize > 0 ? try recv(exactly: payloadSize) : Data()
        return (msgType, payload)
    }

    func disconnect() {
        writeLock.lock()
        defer { writeLock.unlock() }

        if fd >= 0 {
            close(fd)
            fd = -1
        }
    }

    // MARK: - Private

    private func send(data: Data) throws {
        try data.withUnsafeBytes { buffer in
            guard let base = buffer.baseAddress else { return }
            var sent = 0
            while sent < data.count {
                let result = Darwin.send(fd, base + sent, data.count - sent, 0)
                if result < 0 {
                    if errno == EINTR { continue }
                    throw NSError(
                        domain: "SocketClient", code: 4,
                        userInfo: [NSLocalizedDescriptionKey: "Send failed"]
                    )
                }
                guard result > 0 else {
                    throw NSError(
                        domain: "SocketClient", code: 4,
                        userInfo: [NSLocalizedDescriptionKey: "Send failed"]
                    )
                }
                sent += result
            }
        }
    }

    private func recv(exactly count: Int) throws -> Data {
        var buffer = Data(count: count)
        var received = 0
        while received < count {
            let result = buffer.withUnsafeMutableBytes { buf in
                Darwin.recv(fd, buf.baseAddress! + received, count - received, 0)
            }
            if result < 0 {
                if errno == EINTR { continue }
                throw NSError(
                    domain: "SocketClient", code: 5,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Connection closed"
                    ]
                )
            }
            guard result > 0 else {
                throw NSError(
                    domain: "SocketClient", code: 5,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Connection closed"
                    ]
                )
            }
            received += result
        }
        return buffer
    }
}
