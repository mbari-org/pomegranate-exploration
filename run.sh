# Requires defined env var POME_EXPLORATION_DIR pointing to
# the pomegranate-exploration root directory on the host.

if [[ -z ${POME_EXPLORATION_DIR} ]]; then
  echo "POME_EXPLORATION_DIR env var undefined"
  exit
fi

pom_dir=${POME_EXPLORATION_DIR}
wks_dir=`pwd`

# echo "pom_dir=${pom_dir}"
# echo "wks_dir=${wks_dir}"

docker run -it --rm --name pomegranade \
       -v ${pom_dir}:/opt/pomegranate \
       -v ${wks_dir}:/opt/workspace \
       pomegranade:latest \
       $@
