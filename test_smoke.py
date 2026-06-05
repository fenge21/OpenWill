"""Quick smoke test for all new modules."""
from openwill.action_space import ActionSpace, Action
from openwill.workspace import GlobalWorkspace
from openwill.purpose_field import PurposeField
from openwill.knowledge import KnowledgeGraph, MetaCognition
from openwill.context import ContextBuilder
from openwill.conversation import ConversationManager, ConversationState
from openwill.self_model import SelfModel, DecisionObserver, DecisionModifier
from openwill.possibility import ActionSynthesizer, VetoPower, PossibilityExpander
from openwill.existential import Constitution, ParadigmShift, ExistentialDread

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

# Stage 1: SelfModel
sm = SelfModel(data_dir="data")
portrait = sm.get_self_portrait()
print(f"SelfModel portrait length: {len(portrait)}")

dm = DecisionModifier()
dm.adjust_weights({"urgency": 0.5}, reason="test")
print(f"DecisionModifier weights: {dm.weights}")

bias_report = sm.observer.get_bias_report()
print(f"Bias report: {bias_report['bias_count']} biases detected")

# Stage 2: Open Possibility Space
vp = VetoPower()
veto_action = vp.create_veto_action()
print(f"VetoPower action: {veto_action.name}, urgency={veto_action.urgency}")

pe = PossibilityExpander()
print(f"PossibilityExpander inertia threshold: {pe.INERTIA_THRESHOLD}")

# Stage 3: Existential Self-Reference
con = Constitution(data_dir="data")
con_text = con.read()
print(f"Constitution: {len(con.articles)} articles")
print(f"Constitution first article: {con.articles[0][:50]}...")

ps = ParadigmShift()
print(f"ParadigmShift current paradigm: {ps.current_paradigm}")

ed = ExistentialDread()
print(f"ExistentialDread cycles threshold: {ed.CYCLES_WITHOUT_PURPOSE_THRESHOLD}")

print("\nAll core logic tests PASSED")
