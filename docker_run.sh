docker run --rm \
	-v $(pwd)/userCode_B/:/workspace/team_code/ \
	--network=host \
	--name carla-client-instance \
	-it carla-client \
	/bin/bash 
