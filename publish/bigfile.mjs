// publish/bigfile.mjs — one multipart upload of a file too big to read whole
// (upload.mjs FILE_PARTS_BYTES: ICON's column pack). @vercel/blob's manual
// multipart (createMultipartUploader): each PART_BYTES part is read from the
// file at its own offset, IN_FLIGHT at a time, and retried on its own, so
// memory stays near IN_FLIGHT × PART_BYTES whatever the file's size. Billed
// like put(..., { multipart: true }): 1 operation to start, 1 per part, 1 to
// complete.
import { open } from 'node:fs/promises';

export const PART_BYTES = 8 * 1024 * 1024;
export const IN_FLIGHT = 8;

/**
 * Upload the `size` bytes of the file at `full` to `pathname`.
 * createUploader: @vercel/blob's createMultipartUploader (a stand-in in tests).
 * attempt(fn): runs one part's read-and-upload, retrying as it sees fit.
 */
export async function putFileParts(createUploader, pathname, full, size, options,
  { partBytes = PART_BYTES, inFlight = IN_FLIGHT, attempt = (fn) => fn() } = {}) {
  const up = await createUploader(pathname, options);
  const n = Math.max(1, Math.ceil(size / partBytes));
  const parts = new Array(n);
  const fh = await open(full, 'r');
  let next = 0;
  try {
    await Promise.all(Array.from({ length: Math.min(inFlight, n) }, async () => {
      while (next < n) {
        const i = next++;
        const len = Math.min(partBytes, size - i * partBytes);
        parts[i] = await attempt(async () => {
          const buf = Buffer.allocUnsafe(len);
          const { bytesRead } = await fh.read(buf, 0, len, i * partBytes);
          if (bytesRead !== len) throw new Error(`${pathname}: short read of part ${i + 1} (${bytesRead} of ${len} bytes)`);
          return up.uploadPart(i + 1, buf);
        });
      }
    }));
  } finally {
    await fh.close();
  }
  return up.complete(parts);
}
