docker run --rm \
	-v $(pwd)/project/:/workspace/team_code/ \
	--network=host \
	--name carla-client-instance \
	-it carla-client \
	/bin/bash 
