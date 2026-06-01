import Testing
@testable import MemoryStore

@Test func schemaDefinesThreeStatements() {
    #expect(Schema.statements.count == 3)
    #expect(Schema.embeddingDimension > 0)
}
