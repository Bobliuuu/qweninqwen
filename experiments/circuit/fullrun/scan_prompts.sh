#!/bin/bash
# build one flat ELF per candidate single-token prompt from a base cfg.h (prompt differs only in the PROMPT define). usage: scan_prompts.sh BASEDIR OUTDIR TOK...
B=$1; O=$2; shift 2; mkdir -p $O; H=$(cd "$(dirname "$0")" && pwd)
for t in "$@"; do mkdir -p $O/$t; sed "s/^#define PROMPT .*/#define PROMPT {$t}/" $B/cfg.h > $O/$t/cfg.h; $H/build_fw.sh $O/$t $O/$t/flat.elf flat >/dev/null 2>&1; done
