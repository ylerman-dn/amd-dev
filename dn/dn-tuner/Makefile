#
# Drivenets 23/12/2025
#

.DEFAULT_GOAL: build
PLUGIN_SO:=librccl-tunerv4-dn.so
PLUGIN_SO_V5:=librccl-tunerv5-dn.so
include makefiles/common.mk
SRCDIR   ?= $(abspath ../..)
BUILDDIR ?= .
NCCLDIR  := $(BUILDDIR)

SRC_FILES := $(wildcard *.c)

default: ${BUILDDIR}/$(PLUGIN_SO) ${BUILDDIR}/$(PLUGIN_SO_V5)

build: ${BUILDDIR}/$(PLUGIN_SO) ${BUILDDIR}/$(PLUGIN_SO_V5)

${BUILDDIR}/$(PLUGIN_SO): plugin.c
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CC) -Inccl $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

# Same source built with the v5 entry points (init receives RCCL's tuner constants).
${BUILDDIR}/$(PLUGIN_SO_V5): plugin.c nccl/tuner_v5.h
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${BUILDDIR}
	$(CC) -DDN_TUNER_V5 -Inccl $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO_V5) plugin.c

# Host-side (no GPU) test of the v5 port: getCollInfo parity vs v4,
# constants pass-through, override parsing. Runs anywhere.
test: build test_v5_host.c
	$(CC) -Inccl -o ${BUILDDIR}/test_v5_host test_v5_host.c -ldl
	cd ${BUILDDIR} && TEST_RULES_FILE=$(abspath conf/xai_gfx950_tuner.conf) ./test_v5_host

# Optimize configurations from performance data
optimize-config:
	@if [ -z "$(CSV_FILE)" ]; then \
		echo "Usage: make optimize-config CSV_FILE=path/to/data.csv [OUTPUT=config.conf] [METRIC=latency_us]"; \
		echo "Example: make optimize-config CSV_FILE=scripts/sample_performance_data.csv"; \
		exit 1; \
	fi
	python3 scripts/optimize_config.py $(CSV_FILE) \
		$(if $(OUTPUT),-o $(OUTPUT)) \
		$(if $(METRIC),-m $(METRIC)) \
		$(if $(SIZE_RANGES),--size-ranges $(SIZE_RANGES)) \
		$(if $(DRY_RUN),--dry-run) \
		$(if $(NO_HEADER),--no-header)

clean:
	rm -f ${BUILDDIR}/$(PLUGIN_SO) ${BUILDDIR}/$(PLUGIN_SO_V5)

.PHONY: optimize-config clean
