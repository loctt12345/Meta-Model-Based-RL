# Meta-Model-Based-RL

## Mac Setup
1. Install the dependencies  
	pip install -r requirements.txt
2. Download mujoco 131 at https://www.roboti.us/download.html and put it in ~/.mujoco/mjpro131
3. Go to ...site-packages/OpenGL/platform/ctypesloader.py and change this line  
	fullName = util.find_library( name )
to this line  
	fullName = "/System/Library/Frameworks/OpenGL.framework/OpenGL"
