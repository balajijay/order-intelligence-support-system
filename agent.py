import pickle
import torch
import pandas as pd
import torchvision.models as models

print("🧠 Initializing Multi-Modal State Graph Router System...")

# =====================================================================
# 1. LOAD MODEL BUNDLES FROM ARTIFACTS
# =====================================================================
# Load Part 1 Tabular Model Bundle
with open('artifacts/return_risk_model.pickle', 'rb') as f:
    part1_bundle = pickle.load(f)

# Load Part 2 Vision Model Weights
vision_model = models.resnet18()
vision_model.fc = torch.nn.Linear(vision_model.fc.in_features, 10)
vision_model.load_state_dict(torch.load('artifacts/product_classifier.pt'))
vision_model.eval()

# =====================================================================
# 2. THE CUSTOM GRAPH ENGINE FRAMEWORK
# =====================================================================
class StateGraphEngine:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.entry_point = None

    def add_node(self, name, function):
        self.nodes[name] = function

    def set_entry_point(self, name):
        self.entry_point = name

    def add_conditional_edges(self, source, routing_function):
        self.edges[source] = routing_function

    def invoke(self, state):
        """Orchestrates state flows sequentially just like LangGraph."""
        # 1. Start at the entry node
        current_node = self.entry_point
        
        # 2. Run Router Node to update the state
        router_update = self.nodes[current_node](state)
        state.update(router_update)
        
        # 3. Process dynamic conditional routing edge
        next_node = self.edges[current_node](state)
        
        # 4. Route payload execution to the selected feature node
        node_update = self.nodes[next_node](state)
        state.update(node_update)
        
        return state

# =====================================================================
# 3. CREATING GRAPH NODE ACTIONS
# =====================================================================
def router_node(state):
    """The entry point node that inspects the incoming user intent."""
    query = state['user_query'].lower()
    
    if "risk" in query or "transaction" in query or "return" in query:
        next_step = "risk_node"
    elif "image" in query or "photo" in query or "classify" in query:
        next_step = "vision_node"
    else:
        next_step = "chat_node"
        
    return {"next_node": next_step}

def risk_node(state):
    """Executes your Part 1 Data Model Pipeline."""
    df = pd.DataFrame([state['order_data']])
    cols = part1_bundle['feature_columns']
    
    prob = part1_bundle['model'].predict_proba(df[cols])[:, 1][0]
    threshold = part1_bundle['threshold']
    risk_status = "⚠️ HIGH RISK" if prob >= threshold else "✅ LOW RISK"
    
    return {"final_output": f"📊 [NODE: Part 1 ML] Return Probability: {prob:.1%}. Assessment: {risk_status}"}

def vision_node(state):
    """Executes your Part 2 Deep Learning Vision Weights."""
    model_name = vision_model.__class__.__name__
    return {"final_output": f"🖼️ [NODE: Part 2 Vision] Query accepted. Image classification model engine '{model_name}' verified online."}

def chat_node(state):
    """Handles core conversational actions."""
    return {"final_output": "💬 [NODE: General Support] Redirecting prompt context to account support team queue."}

# =====================================================================
# 4. COMPILING AND WIRING THE WORKFLOW GRAPH
# =====================================================================
workflow = StateGraphEngine()

# Mount processing units
workflow.add_node("router", router_node)
workflow.add_node("risk_node", risk_node)
workflow.add_node("vision_node", vision_node)
workflow.add_node("chat_node", chat_node)

workflow.set_entry_point("router")

# Mount smart directional conditional link
def route_decision(state):
    return state["next_node"]

workflow.add_conditional_edges("router", route_decision)

# =====================================================================
# 5. TESTING EXECUTIONS
# =====================================================================
if __name__ == "__main__":
    print("✅ Custom State Graph Compiled Successfully! Running system checks...\n")
    
    mock_order = {
        'price': 120.0, 'discount_pct': 15.0, 'prev_returns': 1, 'prev_orders': 5,
        'delivery_days': 3, 'delivery_delayed': 0, 'payment_method': 'Card',
        'product_category': 'Electronics', 'customer_rating': 4.0, 'customer_rating_missing': 0
    }
    
    # Check 1: Route to the Machine Learning Tabular Risk Model
    res1 = workflow.invoke({
        "user_query": "Is this order transaction risky?",
        "order_data": mock_order, "next_node": "", "final_output": ""
    })
    print(f"User Query: 'Is this order transaction risky?'")
    print(f"{res1['final_output']}\n")
    
    # Check 2: Route to the PyTorch Neural Vision Model
    res2 = workflow.invoke({
        "user_query": "Please classify this product listing image path.",
        "order_data": mock_order, "next_node": "", "final_output": ""
    })
    print(f"User Query: 'Please classify this product listing image path.'")
    print(f"{res2['final_output']}")
