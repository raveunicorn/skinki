import Testing
@testable import SystemBridge

@Test func destructiveCommandRequiresConfirmation() async {
    let runner = ShellRunner(confirm: { _ in false })
    let cmd = ShellRunner.Command(executable: "/bin/rm", arguments: ["-rf", "x"], isDestructive: true)
    await #expect(throws: ShellError.self) {
        _ = try await runner.run(cmd)
    }
}
