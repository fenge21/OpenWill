"""Quick smoke test for all new modules."""
from openwill.action_space import ActionSpace, Action
from openwill.workspace import GlobalWorkspace
from openwill.purpose_field import PurposeField
from openwill.knowledge import KnowledgeGraph, MetaCognition
from openwill.context import ContextBuilder
from openwill.conversation import ConversationManager, ConversationState

# Test ActionSpace
aspace = ActionSpace()
a = Action(name="test", description="test action", urgency=0.5, source="test")
print(f"Action created: {a.name}, urgency={a.urgency}")

# Test GlobalWorkspace
ws = GlobalWorkspace()
ws.submit("curiosity", "test message", 0.8)
winner = ws.resolve()
print(f"Workspace resolve: {winner.source}: {winner.content}")
ctx = ws.get_context_for_llm()
print(f"Workspace context length: {len(ctx)}")

# Test PurposeField
pf = PurposeField(data_dir="data")
pf.add_potential("understand consciousness", 0.7, "curiosity")
dominant = pf.get_dominant()
print(f"PurposeField dominant: {dominant.purpose} ({dominant.strength})")

# Test KnowledgeGraph
kg = KnowledgeGraph(data_dir="data")
kg.add_relation("consciousness", "related_to", "awareness")
stats = kg.get_stats()
print(f"KnowledgeGraph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

# Test ContextBuilder
cb = ContextBuilder()
print(f"ContextBuilder fixed layer: {len(cb._fixed_layer())} chars")

# Test ConversationManager
cm = ConversationManager()
session = cm.get_or_create_session("test")
print(f"Conversation session state: {session.state.value}")

print("All core logic tests PASSED")
