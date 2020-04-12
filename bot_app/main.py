from sanic import Sanic
from sanic.response import json

from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from typing import Dict, Any

import os

def setup_interactive(shared):
    """
    Build and parse CLI opts.
    """
    parser = setup_args()
    parser.add_argument('--port', type=int, default=PORT, help='Port to listen on.')
    SHARED['opt'] = parser.parse_args(print_args=False)

    SHARED['opt']['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'

    # Create model and assign it to the specified task
    agent = create_agent(SHARED.get('opt'), requireModelExists=True)
    SHARED['agent'] = agent
    # SHARED['world'] = create_task(SHARED.get('opt'), SHARED['agent'])

    # show args after loading model
    parser.opt = agent.opt
    parser.print_args()
    return agent.opt

SHARED: Dict[Any, Any] = {}

app = Sanic()

@app.route("/send_message", methods=['POST'])
async def send_message(request):
    world = create_task(SHARED.get('opt'), SHARED['agent'])
    
    return json({"hello": "world"})

if __name__ == "__main__":
    opt = setup_interactive(SHARED)
    app.run(
        host=os.environ.get('HOST', '0.0.0.0'), 
        port=os.environ.get('PORT', 8000)
    )
