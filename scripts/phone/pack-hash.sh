#!/bin/sh
# per-GiB SHA-256 of a file (O_DIRECT reads, so the phone's numbers come from flash, not the page cache): pack-hash.sh FILE > hashes
f=$1; n=$(( ($(stat -c %s "$f") + 1073741823) / 1073741824 ))
i=0; while [ $i -lt $n ]; do
  echo "$i $(dd if="$f" bs=4M skip=$((i * 256)) count=256 iflag=direct status=none | sha256sum | cut -d' ' -f1)"
  i=$((i + 1))
done
