#!/bin/bash
export LOCAL_RANK=$PALS_LOCAL_RANKID
export RANK=$PALS_RANKID
exec "$@"