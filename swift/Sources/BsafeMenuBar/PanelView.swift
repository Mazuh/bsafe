import BsafeMenuKit
import SwiftUI

struct PanelView: View {
    @ObservedObject var controller: BsafeController

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: controller.menuIconName)
                    .foregroundStyle(controller.statusTint)
                Text(controller.statusText)
                    .lineLimit(2)
            }
            .accessibilityLabel("Status: \(controller.statusText)")

            Button(controller.isActive ? "Stop" : "Start") {
                controller.toggleRunning()
            }
            .buttonStyle(.borderedProminent)
            .frame(maxWidth: .infinity)
            .keyboardShortcut(.defaultAction)
            .accessibilityLabel(controller.isActive ? "Stop bsafe" : "Start bsafe")

            Divider()

            Picker("Censor", selection: controller.binding(\.censor)) {
                Text("Body parts").tag(Censor.body)
                Text("All nudity").tag(Censor.all)
                Text("Female").tag(Censor.female)
                Text("Male").tag(Censor.male)
            }
            .pickerStyle(.menu)
            .accessibilityLabel("Censor")

            Toggle("Feet", isOn: controller.feetBinding)
                .disabled(controller.settings.censor == .body)
                .help("Body already includes feet.")
                .accessibilityLabel("Feet")

            Picker("Style", selection: controller.binding(\.style)) {
                Text("Pixelate").tag(Style.pixelate)
                Text("Blur").tag(Style.blur)
                Text("Black box").tag(Style.black)
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Style")

            HStack {
                Text("Padding")
                Slider(value: controller.binding(\.padding), in: 0...0.6, step: 0.1)
                    .accessibilityLabel("Padding")
                Text(formatPadding(controller.settings.padding))
                    .frame(width: 28, alignment: .trailing)
                    .monospacedDigit()
            }

            Stepper(
                "Min padding: \(controller.settings.minPadding) px",
                value: controller.binding(\.minPadding),
                in: 0...64,
                step: 8
            )
            .accessibilityLabel("Min padding")

            Picker("Quality", selection: controller.binding(\.quality)) {
                Text("Fast").tag(Quality.fast)
                Text("Standard").tag(Quality.standard)
                Text("Strict").tag(Quality.strict)
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Quality")
            Text(controller.qualityCaption)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            Divider()

            Toggle(
                "Open at login",
                isOn: Binding(
                    get: { controller.loginEnabled },
                    set: { controller.setLoginEnabled($0) }
                )
            )
            .accessibilityLabel("Open at login")
            if let error = controller.loginError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }

            Button("Quit Bsafe") {
                controller.quit()
            }
            .accessibilityLabel("Quit Bsafe")
        }
        .padding(12)
        .frame(width: 280)
    }
}
