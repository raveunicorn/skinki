import Testing
@testable import TextEngine

@Test func everyActionHasASystemPrompt() {
    for action in TextAction.allCases {
        #expect(!Prompts.system(for: action).isEmpty)
    }
}

@Test func translateMentionsTargetLanguage() {
    let prompt = Prompts.system(for: .translate, targetLanguage: "Russian")
    #expect(prompt.contains("Russian"))
}
