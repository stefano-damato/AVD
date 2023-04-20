docker run --rm ^
	-v %~dp0/carlaContolL6/:/workspace/team_code/ ^
	--network=host ^
	--name carla-client-instance ^
	-it carla-client ^
	/bin/bash