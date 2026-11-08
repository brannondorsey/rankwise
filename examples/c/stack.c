/* Integrate a stack of frames with rankwise from C.
 *
 *   cc -O2 stack.c -I../../rankwise-capi/include -L../../target/release -lrankwise -o stack
 *   ./stack frames.f32 31 4194304 median.f32 clipped.f32 rejected.u16
 *
 * frames.f32 holds n frames of npix float32 values, back to back. Each frame is passed as
 * its own pointer, exactly as a stacker holding registered frames in separate buffers would.
 */
#include <rankwise.h>
#include <stdio.h>
#include <stdlib.h>

static void *slurp(const char *path, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    void *buf = malloc(bytes);
    if (fread(buf, 1, bytes, f) != bytes) { fprintf(stderr, "%s: short read\n", path); exit(1); }
    fclose(f);
    return buf;
}

static void spill(const char *path, const void *buf, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(buf, 1, bytes, f) != bytes) { perror(path); exit(1); }
    fclose(f);
}

#define CHECK(call) do { rw_status s_ = (call); if (s_ != RW_OK) { \
    fprintf(stderr, "%s failed: %s\n", #call, rw_status_string(s_)); return 1; } } while (0)

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s frames.f32 n npix median.f32 clipped.f32 rejected.u16\n", argv[0]);
        return 2;
    }
    size_t n = strtoul(argv[2], NULL, 10), npix = strtoul(argv[3], NULL, 10);
    float *data = slurp(argv[1], n * npix * sizeof(float));
    const float **frames = malloc(n * sizeof *frames);
    for (size_t k = 0; k < n; k++) frames[k] = data + k * npix;   /* one pointer per frame */

    float *median = malloc(npix * sizeof(float));
    float *clipped = malloc(npix * sizeof(float));
    uint16_t *rejected = malloc(npix * sizeof(uint16_t));

    printf("rankwise %s on %s\n", rw_version(), rw_isa());
    CHECK(rw_median_f32(frames, n, npix, median));

    rw_clip clip = rw_clip_sigma(3.0f, 3.0f, /*maxiters*/ 5);   /* astropy sigma_clip semantics */
    CHECK(rw_clip_mean_f32(frames, n, npix, &clip, clipped, rejected));

    spill(argv[4], median, npix * sizeof(float));
    spill(argv[5], clipped, npix * sizeof(float));
    spill(argv[6], rejected, npix * sizeof(uint16_t));
    printf("integrated %zu frames x %zu pixels\n", n, npix);
    free(data); free(frames); free(median); free(clipped); free(rejected);
    return 0;
}
