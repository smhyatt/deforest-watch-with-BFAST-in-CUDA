#pragma once

/*
 * Headers
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>


/*
 * Initialisation
*/

struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new();
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);

/*
 * Arrays
*/

struct futhark_f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr);
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr);
struct futhark_f32_2d ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1);
int futhark_free_f32_2d(struct futhark_context *ctx,
                        struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data);
char *futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                struct futhark_f32_2d *arr);
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr);
struct futhark_i32_1d ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0);
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0);
int futhark_free_i32_1d(struct futhark_context *ctx,
                        struct futhark_i32_1d *arr);
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data);
char *futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                struct futhark_i32_1d *arr);
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_i32_1d **out0,
                       struct futhark_f32_1d **out1, const int32_t in0, const
                       int32_t in1, const int32_t in2, const float in3, const
                       float in4, const float in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
// Start of panic.h.

#include <stdarg.h>

static const char *fut_progname;

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

// End of panic.h.

// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

// End of timing.h.

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent(c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims-1; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    shape[i] = 0;
  }

  return 0;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

// Reading little-endian byte sequences.  On big-endian hosts, we flip
// the resulting bytes.

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_2byte(void* dest) {
  uint16_t x;
  int num_elems_read = fread(&x, 2, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  *(uint16_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_4byte(void* dest) {
  uint32_t x;
  int num_elems_read = fread(&x, 4, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  *(uint32_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_8byte(void* dest) {
  uint64_t x;
  int num_elems_read = fread(&x, 8, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  *(uint64_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int write_byte(void* dest) {
  int num_elems_written = fwrite(dest, 1, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_2byte(void* dest) {
  uint16_t x = *(uint16_t*)dest;
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  int num_elems_written = fwrite(&x, 2, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_4byte(void* dest) {
  uint32_t x = *(uint32_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  int num_elems_written = fwrite(&x, 4, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_8byte(void* dest) {
  uint64_t x = *(uint64_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  int num_elems_written = fwrite(&x, 8, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
  const writer write_bin; // Write in binary format.
  const bin_reader read_bin; // Read in binary format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  uint64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    uint64_t bin_shape;
    ret = read_le_8byte(&bin_shape);
    if (ret != 0) { panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i); }
    elem_count *= bin_shape;
    shape[i] = (int64_t) bin_shape;
  }

  size_t elem_size = expected_type->size;
  void* tmp = realloc(*data, elem_count * elem_size);
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  size_t num_elems_read = fread(*data, elem_size, elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    char* elems = (char*) *data;
    for (uint64_t i=0; i<elem_count; i++) {
      char* elem = elems+(i*elem_size);
      for (unsigned int j=0; j<elem_size/2; j++) {
        char head = elem[j];
        int tail_index = elem_size-1-j;
        elem[j] = elem[tail_index];
        elem[tail_index] = head;
      }
    }
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int64_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 1; i < rank; i++) {
        printf("[]");
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, elem_type->size, num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    return expected_type->read_bin(dest);
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
// Start of tuning.h.

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_size)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      int value = atoi(eql+1);
      if (set_size(cfg, line, value) != 0) {
        strncpy(eql+1, line, max_line_len-strlen(line)-1);
        snprintf(line, max_line_len, "Unknown name '%s' on line %d.", eql+1, lineno);
        return line;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:b", long_options, NULL)) !=
           -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e') {
            if (entry_point != NULL)
                entry_point = optarg;
        }
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s: %s\n", fut_progname,
                    "[-t/--write-runtime-to FILE] [-r/--runs INT] [-D/--debugging] [-L/--log] [-e/--entry-point NAME] [-b/--binary-output]");
            panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
static void futrts_cli_entry_main(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    
    /* Declare and read input. */
    set_binary_mode(stdin);
    
    int32_t read_value_53136;
    
    if (read_scalar(&i32_info, &read_value_53136) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_53137;
    
    if (read_scalar(&i32_info, &read_value_53137) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 1,
              i32_info.type_name, strerror(errno));
    
    int32_t read_value_53138;
    
    if (read_scalar(&i32_info, &read_value_53138) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              i32_info.type_name, strerror(errno));
    
    float read_value_53139;
    
    if (read_scalar(&f32_info, &read_value_53139) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 3,
              f32_info.type_name, strerror(errno));
    
    float read_value_53140;
    
    if (read_scalar(&f32_info, &read_value_53140) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 4,
              f32_info.type_name, strerror(errno));
    
    float read_value_53141;
    
    if (read_scalar(&f32_info, &read_value_53141) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 5,
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *read_value_53142;
    int64_t read_shape_53143[1];
    int32_t *read_arr_53144 = NULL;
    
    errno = 0;
    if (read_array(&i32_info, (void **) &read_arr_53144, read_shape_53143, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 6, "[]",
              i32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_53145;
    int64_t read_shape_53146[2];
    float *read_arr_53147 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_53147, read_shape_53146, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_i32_1d *result_53148;
    struct futhark_f32_1d *result_53149;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        ;
        ;
        ;
        ;
        ;
        ;
        assert((read_value_53142 = futhark_new_i32_1d(ctx, read_arr_53144,
                                                      read_shape_53143[0])) !=
            0);
        assert((read_value_53145 = futhark_new_f32_2d(ctx, read_arr_53147,
                                                      read_shape_53146[0],
                                                      read_shape_53146[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_53148, &result_53149,
                               read_value_53136, read_value_53137,
                               read_value_53138, read_value_53139,
                               read_value_53140, read_value_53141,
                               read_value_53142, read_value_53145);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        ;
        ;
        ;
        ;
        ;
        assert(futhark_free_i32_1d(ctx, read_value_53142) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_53145) == 0);
        assert(futhark_free_i32_1d(ctx, result_53148) == 0);
        assert(futhark_free_f32_1d(ctx, result_53149) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        ;
        ;
        ;
        ;
        ;
        ;
        assert((read_value_53142 = futhark_new_i32_1d(ctx, read_arr_53144,
                                                      read_shape_53143[0])) !=
            0);
        assert((read_value_53145 = futhark_new_f32_2d(ctx, read_arr_53147,
                                                      read_shape_53146[0],
                                                      read_shape_53146[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_main(ctx, &result_53148, &result_53149,
                               read_value_53136, read_value_53137,
                               read_value_53138, read_value_53139,
                               read_value_53140, read_value_53141,
                               read_value_53142, read_value_53145);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        ;
        ;
        ;
        ;
        ;
        assert(futhark_free_i32_1d(ctx, read_value_53142) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_53145) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_i32_1d(ctx, result_53148) == 0);
            assert(futhark_free_f32_1d(ctx, result_53149) == 0);
        }
    }
    ;
    ;
    ;
    ;
    ;
    ;
    free(read_arr_53144);
    free(read_arr_53147);
    if (binary_output)
        set_binary_mode(stdout);
    {
        int32_t *arr = calloc(sizeof(int32_t), futhark_shape_i32_1d(ctx,
                                                                    result_53148)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_i32_1d(ctx, result_53148, arr) == 0);
        write_array(stdout, binary_output, &i32_info, arr,
                    futhark_shape_i32_1d(ctx, result_53148), 1);
        free(arr);
    }
    printf("\n");
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_53149)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_53149, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_53149), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_i32_1d(ctx, result_53148) == 0);
    assert(futhark_free_f32_1d(ctx, result_53149) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="main", .fun =
                                                futrts_cli_entry_main}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    if (entry_point != NULL) {
        int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
        entry_point_fun *entry_point_fun = NULL;
        
        for (int i = 0; i < num_entry_points; i++) {
            if (strcmp(entry_points[i].name, entry_point) == 0) {
                entry_point_fun = entry_points[i].fun;
                break;
            }
        }
        if (entry_point_fun == NULL) {
            fprintf(stderr,
                    "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                    entry_point);
            for (int i = 0; i < num_entry_points; i++)
                fprintf(stderr, "%s\n", entry_points[i].name);
            return 1;
        }
        entry_point_fun(ctx);
        if (runtime_file != NULL)
            fclose(runtime_file);
        futhark_debugging_report(ctx);
    }
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
// Start of lock.h.

/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  (void)lock;
}

#endif

// End of lock.h.

struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
struct futhark_context_config {
    int debugging;
} ;
struct futhark_context_config *futhark_context_config_new()
{
    struct futhark_context_config *cfg =
                                  (struct futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->debugging = 0;
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg);
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int detail)
{
    cfg->debugging = detail;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int detail)
{
    /* Does nothing for this backend. */
    cfg = cfg;
    detail = detail;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
} ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx =
                           (struct futhark_context *) malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->detail_memory = cfg->debugging;
    ctx->debugging = cfg->debugging;
    ctx->error = NULL;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    ctx = ctx;
    return 0;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
static int memblock_unref(struct futhark_context *ctx, struct memblock *block,
                          const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
static int memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                          int64_t size, const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "default space",
              ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocating %lld bytes for %s in %s (then allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    return ret;
}
static int memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                        struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory) {
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->debugging) { }
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock *out_mem_p_53132,
                       int32_t *out_out_arrsizze_53133,
                       struct memblock *out_mem_p_53134,
                       int32_t *out_out_arrsizze_53135,
                       struct memblock mappingindices_mem_52806,
                       struct memblock images_mem_52807, int32_t N_51954,
                       int32_t m_51955, int32_t N_51956, int32_t trend_51957,
                       int32_t k_51958, int32_t n_51959, float freq_51960,
                       float hfrac_51961, float lam_51962);
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline bool itob_i8_bool(int8_t x)
{
    return x;
}
static inline bool itob_i16_bool(int16_t x)
{
    return x;
}
static inline bool itob_i32_bool(int32_t x)
{
    return x;
}
static inline bool itob_i64_bool(int64_t x)
{
    return x;
}
static inline int8_t btoi_bool_i8(bool x)
{
    return x;
}
static inline int16_t btoi_bool_i16(bool x)
{
    return x;
}
static inline int32_t btoi_bool_i32(bool x)
{
    return x;
}
static inline int64_t btoi_bool_i64(bool x)
{
    return x;
}
#define sext_i8_i8(x) ((int8_t) (int8_t) x)
#define sext_i8_i16(x) ((int16_t) (int8_t) x)
#define sext_i8_i32(x) ((int32_t) (int8_t) x)
#define sext_i8_i64(x) ((int64_t) (int8_t) x)
#define sext_i16_i8(x) ((int8_t) (int16_t) x)
#define sext_i16_i16(x) ((int16_t) (int16_t) x)
#define sext_i16_i32(x) ((int32_t) (int16_t) x)
#define sext_i16_i64(x) ((int64_t) (int16_t) x)
#define sext_i32_i8(x) ((int8_t) (int32_t) x)
#define sext_i32_i16(x) ((int16_t) (int32_t) x)
#define sext_i32_i32(x) ((int32_t) (int32_t) x)
#define sext_i32_i64(x) ((int64_t) (int32_t) x)
#define sext_i64_i8(x) ((int8_t) (int64_t) x)
#define sext_i64_i16(x) ((int16_t) (int64_t) x)
#define sext_i64_i32(x) ((int32_t) (int64_t) x)
#define sext_i64_i64(x) ((int64_t) (int64_t) x)
#define zext_i8_i8(x) ((uint8_t) (uint8_t) x)
#define zext_i8_i16(x) ((uint16_t) (uint8_t) x)
#define zext_i8_i32(x) ((uint32_t) (uint8_t) x)
#define zext_i8_i64(x) ((uint64_t) (uint8_t) x)
#define zext_i16_i8(x) ((uint8_t) (uint16_t) x)
#define zext_i16_i16(x) ((uint16_t) (uint16_t) x)
#define zext_i16_i32(x) ((uint32_t) (uint16_t) x)
#define zext_i16_i64(x) ((uint64_t) (uint16_t) x)
#define zext_i32_i8(x) ((uint8_t) (uint32_t) x)
#define zext_i32_i16(x) ((uint16_t) (uint32_t) x)
#define zext_i32_i32(x) ((uint32_t) (uint32_t) x)
#define zext_i32_i64(x) ((uint64_t) (uint32_t) x)
#define zext_i64_i8(x) ((uint8_t) (uint64_t) x)
#define zext_i64_i16(x) ((uint16_t) (uint64_t) x)
#define zext_i64_i32(x) ((uint32_t) (uint64_t) x)
#define zext_i64_i64(x) ((uint64_t) (uint64_t) x)
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return x < y ? x : y;
}
static inline float fmax32(float x, float y)
{
    return x < y ? y : x;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return x < y ? x : y;
}
static inline double fmax64(double x, double y)
{
    return x < y ? y : x;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_gamma32(float x)
{
    return tgamma(x);
}
static inline float futrts_lgamma32(float x)
{
    return lgamma(x);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline char futrts_isnan32(float x)
{
    return isnan(x);
}
static inline char futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_gamma64(double x)
{
    return tgamma(x);
}
static inline double futrts_lgamma64(double x)
{
    return lgamma(x);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline char futrts_isnan64(double x)
{
    return isnan(x);
}
static inline char futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static int futrts_main(struct futhark_context *ctx,
                       struct memblock *out_mem_p_53132,
                       int32_t *out_out_arrsizze_53133,
                       struct memblock *out_mem_p_53134,
                       int32_t *out_out_arrsizze_53135,
                       struct memblock mappingindices_mem_52806,
                       struct memblock images_mem_52807, int32_t N_51954,
                       int32_t m_51955, int32_t N_51956, int32_t trend_51957,
                       int32_t k_51958, int32_t n_51959, float freq_51960,
                       float hfrac_51961, float lam_51962)
{
    struct memblock out_mem_53074;
    
    out_mem_53074.references = NULL;
    
    int32_t out_arrsizze_53075;
    struct memblock out_mem_53076;
    
    out_mem_53076.references = NULL;
    
    int32_t out_arrsizze_53077;
    bool dim_zzero_51965 = 0 == m_51955;
    bool dim_zzero_51966 = 0 == N_51956;
    bool old_empty_51967 = dim_zzero_51965 || dim_zzero_51966;
    bool dim_zzero_51968 = 0 == N_51954;
    bool new_empty_51969 = dim_zzero_51965 || dim_zzero_51968;
    bool both_empty_51970 = old_empty_51967 && new_empty_51969;
    bool dim_match_51971 = N_51954 == N_51956;
    bool empty_or_match_51972 = both_empty_51970 || dim_match_51971;
    bool empty_or_match_cert_51973;
    
    if (!empty_or_match_51972) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "bfast-irreg.fut:131:1-282:20",
                               "function arguments of wrong shape");
        if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
            return 1;
        return 1;
    }
    
    int32_t x_51975 = 2 * k_51958;
    int32_t k2p2_51976 = 2 + x_51975;
    bool cond_51977 = slt32(0, trend_51957);
    int32_t k2p2zq_51978;
    
    if (cond_51977) {
        k2p2zq_51978 = k2p2_51976;
    } else {
        int32_t res_51979 = k2p2_51976 - 1;
        
        k2p2zq_51978 = res_51979;
    }
    
    int64_t binop_x_52809 = sext_i32_i64(k2p2zq_51978);
    int64_t binop_y_52810 = sext_i32_i64(N_51954);
    int64_t binop_x_52811 = binop_x_52809 * binop_y_52810;
    int64_t bytes_52808 = 4 * binop_x_52811;
    int64_t binop_x_52824 = sext_i32_i64(k2p2zq_51978);
    int64_t binop_y_52825 = sext_i32_i64(N_51954);
    int64_t binop_x_52826 = binop_x_52824 * binop_y_52825;
    int64_t bytes_52823 = 4 * binop_x_52826;
    struct memblock lifted_1_zlzb_arg_mem_52838;
    
    lifted_1_zlzb_arg_mem_52838.references = NULL;
    if (cond_51977) {
        bool bounds_invalid_upwards_51981 = slt32(k2p2zq_51978, 0);
        bool eq_x_zz_51982 = 0 == k2p2zq_51978;
        bool not_p_51983 = !bounds_invalid_upwards_51981;
        bool p_and_eq_x_y_51984 = eq_x_zz_51982 && not_p_51983;
        bool dim_zzero_51985 = bounds_invalid_upwards_51981 ||
             p_and_eq_x_y_51984;
        bool both_empty_51986 = eq_x_zz_51982 && dim_zzero_51985;
        bool empty_or_match_51990 = not_p_51983 || both_empty_51986;
        bool empty_or_match_cert_51991;
        
        if (!empty_or_match_51990) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:131:1-282:20 -> bfast-irreg.fut:142:16-55 -> bfast-irreg.fut:62:10-18 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_51978, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                               "lifted_1_zlzb_arg_mem_52838") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_52812;
        
        mem_52812.references = NULL;
        if (memblock_alloc(ctx, &mem_52812, bytes_52808, "mem_52812"))
            return 1;
        for (int32_t i_52493 = 0; i_52493 < k2p2zq_51978; i_52493++) {
            bool cond_51995 = i_52493 == 0;
            bool cond_51996 = i_52493 == 1;
            int32_t r32_arg_51997 = sdiv32(i_52493, 2);
            int32_t x_51998 = smod32(i_52493, 2);
            float res_51999 = sitofp_i32_f32(r32_arg_51997);
            bool cond_52000 = x_51998 == 0;
            float x_52001 = 6.2831855F * res_51999;
            
            for (int32_t i_52489 = 0; i_52489 < N_51954; i_52489++) {
                int32_t x_52003 =
                        ((int32_t *) mappingindices_mem_52806.mem)[i_52489];
                float res_52004;
                
                if (cond_51995) {
                    res_52004 = 1.0F;
                } else {
                    float res_52005;
                    
                    if (cond_51996) {
                        float res_52006 = sitofp_i32_f32(x_52003);
                        
                        res_52005 = res_52006;
                    } else {
                        float res_52007 = sitofp_i32_f32(x_52003);
                        float x_52008 = x_52001 * res_52007;
                        float angle_52009 = x_52008 / freq_51960;
                        float res_52010;
                        
                        if (cond_52000) {
                            float res_52011;
                            
                            res_52011 = futrts_sin32(angle_52009);
                            res_52010 = res_52011;
                        } else {
                            float res_52012;
                            
                            res_52012 = futrts_cos32(angle_52009);
                            res_52010 = res_52012;
                        }
                        res_52005 = res_52010;
                    }
                    res_52004 = res_52005;
                }
                ((float *) mem_52812.mem)[i_52493 * N_51954 + i_52489] =
                    res_52004;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_52838, &mem_52812,
                         "mem_52812") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52812, "mem_52812") != 0)
            return 1;
    } else {
        bool bounds_invalid_upwards_52013 = slt32(k2p2zq_51978, 0);
        bool eq_x_zz_52014 = 0 == k2p2zq_51978;
        bool not_p_52015 = !bounds_invalid_upwards_52013;
        bool p_and_eq_x_y_52016 = eq_x_zz_52014 && not_p_52015;
        bool dim_zzero_52017 = bounds_invalid_upwards_52013 ||
             p_and_eq_x_y_52016;
        bool both_empty_52018 = eq_x_zz_52014 && dim_zzero_52017;
        bool empty_or_match_52022 = not_p_52015 || both_empty_52018;
        bool empty_or_match_cert_52023;
        
        if (!empty_or_match_52022) {
            ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                                   "bfast-irreg.fut:131:1-282:20 -> bfast-irreg.fut:143:16-55 -> bfast-irreg.fut:74:10-20 -> /futlib/array.fut:61:1-62:12",
                                   "Function return value does not match shape of type ",
                                   "*", "[", k2p2zq_51978, "]",
                                   "intrinsics.i32");
            if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                               "lifted_1_zlzb_arg_mem_52838") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
                return 1;
            if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
                return 1;
            return 1;
        }
        
        struct memblock mem_52827;
        
        mem_52827.references = NULL;
        if (memblock_alloc(ctx, &mem_52827, bytes_52823, "mem_52827"))
            return 1;
        for (int32_t i_52501 = 0; i_52501 < k2p2zq_51978; i_52501++) {
            bool cond_52027 = i_52501 == 0;
            int32_t i_52028 = 1 + i_52501;
            int32_t r32_arg_52029 = sdiv32(i_52028, 2);
            int32_t x_52030 = smod32(i_52028, 2);
            float res_52031 = sitofp_i32_f32(r32_arg_52029);
            bool cond_52032 = x_52030 == 0;
            float x_52033 = 6.2831855F * res_52031;
            
            for (int32_t i_52497 = 0; i_52497 < N_51954; i_52497++) {
                int32_t x_52035 =
                        ((int32_t *) mappingindices_mem_52806.mem)[i_52497];
                float res_52036;
                
                if (cond_52027) {
                    res_52036 = 1.0F;
                } else {
                    float res_52037 = sitofp_i32_f32(x_52035);
                    float x_52038 = x_52033 * res_52037;
                    float angle_52039 = x_52038 / freq_51960;
                    float res_52040;
                    
                    if (cond_52032) {
                        float res_52041;
                        
                        res_52041 = futrts_sin32(angle_52039);
                        res_52040 = res_52041;
                    } else {
                        float res_52042;
                        
                        res_52042 = futrts_cos32(angle_52039);
                        res_52040 = res_52042;
                    }
                    res_52036 = res_52040;
                }
                ((float *) mem_52827.mem)[i_52501 * N_51954 + i_52497] =
                    res_52036;
            }
        }
        if (memblock_set(ctx, &lifted_1_zlzb_arg_mem_52838, &mem_52827,
                         "mem_52827") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52827, "mem_52827") != 0)
            return 1;
    }
    
    int32_t x_52044 = N_51954 * N_51954;
    int32_t y_52045 = 2 * N_51954;
    int32_t x_52046 = x_52044 + y_52045;
    int32_t x_52047 = 1 + x_52046;
    int32_t y_52048 = 1 + N_51954;
    int32_t x_52049 = sdiv32(x_52047, y_52048);
    int32_t x_52050 = x_52049 - N_51954;
    int32_t lifted_1_zlzb_arg_52051 = x_52050 - 1;
    float res_52052 = sitofp_i32_f32(lifted_1_zlzb_arg_52051);
    int64_t binop_x_52840 = sext_i32_i64(N_51954);
    int64_t binop_y_52841 = sext_i32_i64(k2p2zq_51978);
    int64_t binop_x_52842 = binop_x_52840 * binop_y_52841;
    int64_t bytes_52839 = 4 * binop_x_52842;
    struct memblock mem_52843;
    
    mem_52843.references = NULL;
    if (memblock_alloc(ctx, &mem_52843, bytes_52839, "mem_52843"))
        return 1;
    for (int32_t i_52509 = 0; i_52509 < N_51954; i_52509++) {
        for (int32_t i_52505 = 0; i_52505 < k2p2zq_51978; i_52505++) {
            float x_52057 =
                  ((float *) lifted_1_zlzb_arg_mem_52838.mem)[i_52505 *
                                                              N_51954 +
                                                              i_52509];
            float res_52058 = res_52052 + x_52057;
            
            ((float *) mem_52843.mem)[i_52509 * k2p2zq_51978 + i_52505] =
                res_52058;
        }
    }
    
    int32_t m_52061 = k2p2zq_51978 - 1;
    bool empty_slice_52068 = n_51959 == 0;
    int32_t m_52069 = n_51959 - 1;
    bool zzero_leq_i_p_m_t_s_52070 = sle32(0, m_52069);
    bool i_p_m_t_s_leq_w_52071 = slt32(m_52069, N_51954);
    bool i_lte_j_52072 = sle32(0, n_51959);
    bool y_52073 = zzero_leq_i_p_m_t_s_52070 && i_p_m_t_s_leq_w_52071;
    bool y_52074 = i_lte_j_52072 && y_52073;
    bool ok_or_empty_52075 = empty_slice_52068 || y_52074;
    bool index_certs_52077;
    
    if (!ok_or_empty_52075) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:131:1-282:20 -> bfast-irreg.fut:152:15-21",
                               "Index [", 0, ", ", "", ":", n_51959,
                               "] out of bounds for array of shape [",
                               k2p2zq_51978, "][", N_51954, "].");
        if (memblock_unref(ctx, &mem_52843, "mem_52843") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                           "lifted_1_zlzb_arg_mem_52838") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_52079;
    
    if (!ok_or_empty_52075) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:131:1-282:20 -> bfast-irreg.fut:153:15-22",
                               "Index [", "", ":", n_51959, ", ", 0,
                               "] out of bounds for array of shape [", N_51954,
                               "][", k2p2zq_51978, "].");
        if (memblock_unref(ctx, &mem_52843, "mem_52843") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                           "lifted_1_zlzb_arg_mem_52838") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
            return 1;
        return 1;
    }
    
    bool index_certs_52090;
    
    if (!ok_or_empty_52075) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%s%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:131:1-282:20 -> bfast-irreg.fut:154:15-26",
                               "Index [", 0, ", ", "", ":", n_51959,
                               "] out of bounds for array of shape [", m_51955,
                               "][", N_51954, "].");
        if (memblock_unref(ctx, &mem_52843, "mem_52843") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                           "lifted_1_zlzb_arg_mem_52838") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_x_52855 = sext_i32_i64(m_51955);
    int64_t binop_x_52857 = binop_y_52841 * binop_x_52855;
    int64_t binop_x_52859 = binop_y_52841 * binop_x_52857;
    int64_t bytes_52854 = 4 * binop_x_52859;
    struct memblock mem_52860;
    
    mem_52860.references = NULL;
    if (memblock_alloc(ctx, &mem_52860, bytes_52854, "mem_52860"))
        return 1;
    for (int32_t i_52523 = 0; i_52523 < m_51955; i_52523++) {
        for (int32_t i_52519 = 0; i_52519 < k2p2zq_51978; i_52519++) {
            for (int32_t i_52515 = 0; i_52515 < k2p2zq_51978; i_52515++) {
                float res_52099;
                float redout_52511 = 0.0F;
                
                for (int32_t i_52512 = 0; i_52512 < n_51959; i_52512++) {
                    float x_52103 = ((float *) images_mem_52807.mem)[i_52523 *
                                                                     N_51956 +
                                                                     i_52512];
                    float x_52104 =
                          ((float *) lifted_1_zlzb_arg_mem_52838.mem)[i_52519 *
                                                                      N_51954 +
                                                                      i_52512];
                    float x_52105 = ((float *) mem_52843.mem)[i_52512 *
                                                              k2p2zq_51978 +
                                                              i_52515];
                    float x_52106 = x_52104 * x_52105;
                    bool res_52107;
                    
                    res_52107 = futrts_isnan32(x_52103);
                    
                    float y_52108;
                    
                    if (res_52107) {
                        y_52108 = 0.0F;
                    } else {
                        y_52108 = 1.0F;
                    }
                    
                    float res_52109 = x_52106 * y_52108;
                    float res_52102 = res_52109 + redout_52511;
                    float redout_tmp_53087 = res_52102;
                    
                    redout_52511 = redout_tmp_53087;
                }
                res_52099 = redout_52511;
                ((float *) mem_52860.mem)[i_52523 * (k2p2zq_51978 *
                                                     k2p2zq_51978) + i_52519 *
                                          k2p2zq_51978 + i_52515] = res_52099;
            }
        }
    }
    
    int32_t j_52111 = 2 * k2p2zq_51978;
    int32_t j_m_i_52112 = j_52111 - k2p2zq_51978;
    int32_t nm_52115 = k2p2zq_51978 * j_52111;
    bool empty_slice_52128 = j_m_i_52112 == 0;
    int32_t m_52129 = j_m_i_52112 - 1;
    int32_t i_p_m_t_s_52130 = k2p2zq_51978 + m_52129;
    bool zzero_leq_i_p_m_t_s_52131 = sle32(0, i_p_m_t_s_52130);
    bool ok_or_empty_52138 = empty_slice_52128 || zzero_leq_i_p_m_t_s_52131;
    bool index_certs_52140;
    
    if (!ok_or_empty_52138) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s%d%s%d%s%d%s%d%s\n",
                               "bfast-irreg.fut:131:1-282:20 -> bfast-irreg.fut:166:14-29 -> bfast-irreg.fut:107:8-37",
                               "Index [", 0, ":", k2p2zq_51978, ", ",
                               k2p2zq_51978, ":", j_52111,
                               "] out of bounds for array of shape [",
                               k2p2zq_51978, "][", j_52111, "].");
        if (memblock_unref(ctx, &mem_52860, "mem_52860") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52843, "mem_52843") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                           "lifted_1_zlzb_arg_mem_52838") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
            return 1;
        return 1;
    }
    
    int64_t binop_y_52892 = sext_i32_i64(j_m_i_52112);
    int64_t binop_x_52893 = binop_x_52857 * binop_y_52892;
    int64_t bytes_52888 = 4 * binop_x_52893;
    struct memblock mem_52894;
    
    mem_52894.references = NULL;
    if (memblock_alloc(ctx, &mem_52894, bytes_52888, "mem_52894"))
        return 1;
    
    int64_t binop_x_52897 = sext_i32_i64(nm_52115);
    int64_t bytes_52896 = 4 * binop_x_52897;
    struct memblock mem_52898;
    
    mem_52898.references = NULL;
    if (memblock_alloc(ctx, &mem_52898, bytes_52896, "mem_52898"))
        return 1;
    
    struct memblock mem_52904;
    
    mem_52904.references = NULL;
    if (memblock_alloc(ctx, &mem_52904, bytes_52896, "mem_52904"))
        return 1;
    for (int32_t i_52545 = 0; i_52545 < m_51955; i_52545++) {
        for (int32_t i_52527 = 0; i_52527 < nm_52115; i_52527++) {
            int32_t res_52145 = sdiv32(i_52527, j_52111);
            int32_t res_52146 = smod32(i_52527, j_52111);
            bool cond_52147 = slt32(res_52146, k2p2zq_51978);
            float res_52148;
            
            if (cond_52147) {
                float res_52149 = ((float *) mem_52860.mem)[i_52545 *
                                                            (k2p2zq_51978 *
                                                             k2p2zq_51978) +
                                                            res_52145 *
                                                            k2p2zq_51978 +
                                                            res_52146];
                
                res_52148 = res_52149;
            } else {
                int32_t y_52150 = k2p2zq_51978 + res_52145;
                bool cond_52151 = res_52146 == y_52150;
                float res_52152;
                
                if (cond_52151) {
                    res_52152 = 1.0F;
                } else {
                    res_52152 = 0.0F;
                }
                res_52148 = res_52152;
            }
            ((float *) mem_52898.mem)[i_52527] = res_52148;
        }
        for (int32_t i_52155 = 0; i_52155 < k2p2zq_51978; i_52155++) {
            float v1_52160 = ((float *) mem_52898.mem)[i_52155];
            bool cond_52161 = v1_52160 == 0.0F;
            
            for (int32_t i_52531 = 0; i_52531 < nm_52115; i_52531++) {
                int32_t res_52164 = sdiv32(i_52531, j_52111);
                int32_t res_52165 = smod32(i_52531, j_52111);
                float res_52166;
                
                if (cond_52161) {
                    int32_t x_52167 = j_52111 * res_52164;
                    int32_t i_52168 = res_52165 + x_52167;
                    float res_52169 = ((float *) mem_52898.mem)[i_52168];
                    
                    res_52166 = res_52169;
                } else {
                    float x_52170 = ((float *) mem_52898.mem)[res_52165];
                    float x_52171 = x_52170 / v1_52160;
                    bool cond_52172 = slt32(res_52164, m_52061);
                    float res_52173;
                    
                    if (cond_52172) {
                        int32_t x_52174 = 1 + res_52164;
                        int32_t x_52175 = j_52111 * x_52174;
                        int32_t i_52176 = res_52165 + x_52175;
                        float x_52177 = ((float *) mem_52898.mem)[i_52176];
                        int32_t i_52178 = i_52155 + x_52175;
                        float x_52179 = ((float *) mem_52898.mem)[i_52178];
                        float y_52180 = x_52171 * x_52179;
                        float res_52181 = x_52177 - y_52180;
                        
                        res_52173 = res_52181;
                    } else {
                        res_52173 = x_52171;
                    }
                    res_52166 = res_52173;
                }
                ((float *) mem_52904.mem)[i_52531] = res_52166;
            }
            for (int32_t write_iter_52533 = 0; write_iter_52533 < nm_52115;
                 write_iter_52533++) {
                bool less_than_zzero_52537 = slt32(write_iter_52533, 0);
                bool greater_than_sizze_52538 = sle32(nm_52115,
                                                      write_iter_52533);
                bool outside_bounds_dim_52539 = less_than_zzero_52537 ||
                     greater_than_sizze_52538;
                
                if (!outside_bounds_dim_52539) {
                    memmove(mem_52898.mem + write_iter_52533 * 4,
                            mem_52904.mem + write_iter_52533 * 4,
                            sizeof(float));
                }
            }
        }
        for (int32_t i_53093 = 0; i_53093 < k2p2zq_51978; i_53093++) {
            for (int32_t i_53094 = 0; i_53094 < j_m_i_52112; i_53094++) {
                ((float *) mem_52894.mem)[i_52545 * (j_m_i_52112 *
                                                     k2p2zq_51978) + (i_53093 *
                                                                      j_m_i_52112 +
                                                                      i_53094)] =
                    ((float *) mem_52898.mem)[k2p2zq_51978 + (i_53093 *
                                                              j_52111 +
                                                              i_53094)];
            }
        }
    }
    if (memblock_unref(ctx, &mem_52860, "mem_52860") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52898, "mem_52898") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52904, "mem_52904") != 0)
        return 1;
    
    int64_t bytes_52911 = 4 * binop_x_52857;
    struct memblock mem_52915;
    
    mem_52915.references = NULL;
    if (memblock_alloc(ctx, &mem_52915, bytes_52911, "mem_52915"))
        return 1;
    for (int32_t i_52555 = 0; i_52555 < m_51955; i_52555++) {
        for (int32_t i_52551 = 0; i_52551 < k2p2zq_51978; i_52551++) {
            float res_52192;
            float redout_52547 = 0.0F;
            
            for (int32_t i_52548 = 0; i_52548 < n_51959; i_52548++) {
                float x_52196 =
                      ((float *) lifted_1_zlzb_arg_mem_52838.mem)[i_52551 *
                                                                  N_51954 +
                                                                  i_52548];
                float x_52197 = ((float *) images_mem_52807.mem)[i_52555 *
                                                                 N_51956 +
                                                                 i_52548];
                bool res_52198;
                
                res_52198 = futrts_isnan32(x_52197);
                
                float res_52199;
                
                if (res_52198) {
                    res_52199 = 0.0F;
                } else {
                    float res_52200 = x_52196 * x_52197;
                    
                    res_52199 = res_52200;
                }
                
                float res_52195 = res_52199 + redout_52547;
                float redout_tmp_53097 = res_52195;
                
                redout_52547 = redout_tmp_53097;
            }
            res_52192 = redout_52547;
            ((float *) mem_52915.mem)[i_52555 * k2p2zq_51978 + i_52551] =
                res_52192;
        }
    }
    if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                       "lifted_1_zlzb_arg_mem_52838") != 0)
        return 1;
    
    struct memblock mem_52930;
    
    mem_52930.references = NULL;
    if (memblock_alloc(ctx, &mem_52930, bytes_52911, "mem_52930"))
        return 1;
    for (int32_t i_52565 = 0; i_52565 < m_51955; i_52565++) {
        for (int32_t i_52561 = 0; i_52561 < k2p2zq_51978; i_52561++) {
            float res_52212;
            float redout_52557 = 0.0F;
            
            for (int32_t i_52558 = 0; i_52558 < j_m_i_52112; i_52558++) {
                float x_52216 = ((float *) mem_52915.mem)[i_52565 *
                                                          k2p2zq_51978 +
                                                          i_52558];
                float x_52217 = ((float *) mem_52894.mem)[i_52565 *
                                                          (j_m_i_52112 *
                                                           k2p2zq_51978) +
                                                          i_52561 *
                                                          j_m_i_52112 +
                                                          i_52558];
                float res_52218 = x_52216 * x_52217;
                float res_52215 = res_52218 + redout_52557;
                float redout_tmp_53100 = res_52215;
                
                redout_52557 = redout_tmp_53100;
            }
            res_52212 = redout_52557;
            ((float *) mem_52930.mem)[i_52565 * k2p2zq_51978 + i_52561] =
                res_52212;
        }
    }
    if (memblock_unref(ctx, &mem_52894, "mem_52894") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52915, "mem_52915") != 0)
        return 1;
    
    int64_t binop_x_52944 = binop_x_52840 * binop_x_52855;
    int64_t bytes_52941 = 4 * binop_x_52944;
    struct memblock mem_52945;
    
    mem_52945.references = NULL;
    if (memblock_alloc(ctx, &mem_52945, bytes_52941, "mem_52945"))
        return 1;
    for (int32_t i_52575 = 0; i_52575 < m_51955; i_52575++) {
        for (int32_t i_52571 = 0; i_52571 < N_51954; i_52571++) {
            float res_52224;
            float redout_52567 = 0.0F;
            
            for (int32_t i_52568 = 0; i_52568 < k2p2zq_51978; i_52568++) {
                float x_52228 = ((float *) mem_52930.mem)[i_52575 *
                                                          k2p2zq_51978 +
                                                          i_52568];
                float x_52229 = ((float *) mem_52843.mem)[i_52571 *
                                                          k2p2zq_51978 +
                                                          i_52568];
                float res_52230 = x_52228 * x_52229;
                float res_52227 = res_52230 + redout_52567;
                float redout_tmp_53103 = res_52227;
                
                redout_52567 = redout_tmp_53103;
            }
            res_52224 = redout_52567;
            ((float *) mem_52945.mem)[i_52575 * N_51954 + i_52571] = res_52224;
        }
    }
    if (memblock_unref(ctx, &mem_52843, "mem_52843") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52930, "mem_52930") != 0)
        return 1;
    
    int32_t i_52232 = N_51954 - 1;
    bool x_52233 = sle32(0, i_52232);
    bool index_certs_52236;
    
    if (!x_52233) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                               "bfast-irreg.fut:131:1-282:20 -> bfast-irreg.fut:187:5-196:25 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> bfast-irreg.fut:192:30-91 -> bfast-irreg.fut:35:13-20 -> /futlib/array.fut:18:29-34",
                               "Index [", i_52232,
                               "] out of bounds for array of shape [", N_51954,
                               "].");
        if (memblock_unref(ctx, &mem_52945, "mem_52945") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52930, "mem_52930") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52915, "mem_52915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52904, "mem_52904") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52898, "mem_52898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52894, "mem_52894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52860, "mem_52860") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52843, "mem_52843") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                           "lifted_1_zlzb_arg_mem_52838") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
            return 1;
        return 1;
    }
    
    int64_t bytes_52956 = 4 * binop_x_52855;
    struct memblock mem_52958;
    
    mem_52958.references = NULL;
    if (memblock_alloc(ctx, &mem_52958, bytes_52956, "mem_52958"))
        return 1;
    
    struct memblock mem_52963;
    
    mem_52963.references = NULL;
    if (memblock_alloc(ctx, &mem_52963, bytes_52941, "mem_52963"))
        return 1;
    
    struct memblock mem_52968;
    
    mem_52968.references = NULL;
    if (memblock_alloc(ctx, &mem_52968, bytes_52941, "mem_52968"))
        return 1;
    
    int64_t bytes_52972 = 4 * binop_x_52840;
    struct memblock mem_52974;
    
    mem_52974.references = NULL;
    if (memblock_alloc(ctx, &mem_52974, bytes_52972, "mem_52974"))
        return 1;
    
    struct memblock mem_52977;
    
    mem_52977.references = NULL;
    if (memblock_alloc(ctx, &mem_52977, bytes_52972, "mem_52977"))
        return 1;
    
    struct memblock mem_52984;
    
    mem_52984.references = NULL;
    if (memblock_alloc(ctx, &mem_52984, bytes_52972, "mem_52984"))
        return 1;
    for (int32_t i_53104 = 0; i_53104 < N_51954; i_53104++) {
        ((float *) mem_52984.mem)[i_53104] = NAN;
    }
    
    struct memblock mem_52987;
    
    mem_52987.references = NULL;
    if (memblock_alloc(ctx, &mem_52987, bytes_52972, "mem_52987"))
        return 1;
    for (int32_t i_53105 = 0; i_53105 < N_51954; i_53105++) {
        ((int32_t *) mem_52987.mem)[i_53105] = 0;
    }
    
    struct memblock mem_52992;
    
    mem_52992.references = NULL;
    if (memblock_alloc(ctx, &mem_52992, bytes_52972, "mem_52992"))
        return 1;
    
    struct memblock mem_53002;
    
    mem_53002.references = NULL;
    if (memblock_alloc(ctx, &mem_53002, bytes_52972, "mem_53002"))
        return 1;
    for (int32_t i_52610 = 0; i_52610 < m_51955; i_52610++) {
        int32_t discard_52585;
        int32_t scanacc_52579 = 0;
        
        for (int32_t i_52582 = 0; i_52582 < N_51954; i_52582++) {
            float x_52266 = ((float *) images_mem_52807.mem)[i_52610 * N_51956 +
                                                             i_52582];
            float x_52267 = ((float *) mem_52945.mem)[i_52610 * N_51954 +
                                                      i_52582];
            bool res_52268;
            
            res_52268 = futrts_isnan32(x_52266);
            
            bool cond_52269 = !res_52268;
            float res_52270;
            
            if (cond_52269) {
                float res_52271 = x_52266 - x_52267;
                
                res_52270 = res_52271;
            } else {
                res_52270 = NAN;
            }
            
            bool res_52272;
            
            res_52272 = futrts_isnan32(res_52270);
            
            bool res_52273 = !res_52272;
            int32_t res_52274;
            
            if (res_52273) {
                res_52274 = 1;
            } else {
                res_52274 = 0;
            }
            
            int32_t res_52265 = res_52274 + scanacc_52579;
            
            ((int32_t *) mem_52974.mem)[i_52582] = res_52265;
            ((float *) mem_52977.mem)[i_52582] = res_52270;
            
            int32_t scanacc_tmp_53109 = res_52265;
            
            scanacc_52579 = scanacc_tmp_53109;
        }
        discard_52585 = scanacc_52579;
        memmove(mem_52963.mem + i_52610 * N_51954 * 4, mem_52984.mem + 0,
                N_51954 * sizeof(float));
        memmove(mem_52968.mem + i_52610 * N_51954 * 4, mem_52987.mem + 0,
                N_51954 * sizeof(int32_t));
        for (int32_t write_iter_52586 = 0; write_iter_52586 < N_51954;
             write_iter_52586++) {
            float write_iv_52589 = ((float *) mem_52977.mem)[write_iter_52586];
            int32_t write_iv_52590 =
                    ((int32_t *) mem_52974.mem)[write_iter_52586];
            bool res_52285;
            
            res_52285 = futrts_isnan32(write_iv_52589);
            
            bool res_52286 = !res_52285;
            int32_t res_52287;
            
            if (res_52286) {
                int32_t res_52288 = write_iv_52590 - 1;
                
                res_52287 = res_52288;
            } else {
                res_52287 = -1;
            }
            
            bool less_than_zzero_52592 = slt32(res_52287, 0);
            bool greater_than_sizze_52593 = sle32(N_51954, res_52287);
            bool outside_bounds_dim_52594 = less_than_zzero_52592 ||
                 greater_than_sizze_52593;
            
            memmove(mem_52992.mem + 0, mem_52968.mem + i_52610 * N_51954 * 4,
                    N_51954 * sizeof(int32_t));
            
            struct memblock write_out_mem_52999;
            
            write_out_mem_52999.references = NULL;
            if (outside_bounds_dim_52594) {
                if (memblock_set(ctx, &write_out_mem_52999, &mem_52992,
                                 "mem_52992") != 0)
                    return 1;
            } else {
                struct memblock mem_52995;
                
                mem_52995.references = NULL;
                if (memblock_alloc(ctx, &mem_52995, 4, "mem_52995"))
                    return 1;
                
                int32_t x_53115;
                
                for (int32_t i_53114 = 0; i_53114 < 1; i_53114++) {
                    x_53115 = write_iter_52586 + sext_i32_i32(i_53114);
                    ((int32_t *) mem_52995.mem)[i_53114] = x_53115;
                }
                
                struct memblock mem_52998;
                
                mem_52998.references = NULL;
                if (memblock_alloc(ctx, &mem_52998, bytes_52972, "mem_52998"))
                    return 1;
                memmove(mem_52998.mem + 0, mem_52968.mem + i_52610 * N_51954 *
                        4, N_51954 * sizeof(int32_t));
                memmove(mem_52998.mem + res_52287 * 4, mem_52995.mem + 0,
                        sizeof(int32_t));
                if (memblock_unref(ctx, &mem_52995, "mem_52995") != 0)
                    return 1;
                if (memblock_set(ctx, &write_out_mem_52999, &mem_52998,
                                 "mem_52998") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_52998, "mem_52998") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_52995, "mem_52995") != 0)
                    return 1;
            }
            memmove(mem_52968.mem + i_52610 * N_51954 * 4,
                    write_out_mem_52999.mem + 0, N_51954 * sizeof(int32_t));
            if (memblock_unref(ctx, &write_out_mem_52999,
                               "write_out_mem_52999") != 0)
                return 1;
            memmove(mem_53002.mem + 0, mem_52963.mem + i_52610 * N_51954 * 4,
                    N_51954 * sizeof(float));
            
            struct memblock write_out_mem_53006;
            
            write_out_mem_53006.references = NULL;
            if (outside_bounds_dim_52594) {
                if (memblock_set(ctx, &write_out_mem_53006, &mem_53002,
                                 "mem_53002") != 0)
                    return 1;
            } else {
                struct memblock mem_53005;
                
                mem_53005.references = NULL;
                if (memblock_alloc(ctx, &mem_53005, bytes_52972, "mem_53005"))
                    return 1;
                memmove(mem_53005.mem + 0, mem_52963.mem + i_52610 * N_51954 *
                        4, N_51954 * sizeof(float));
                memmove(mem_53005.mem + res_52287 * 4, mem_52977.mem +
                        write_iter_52586 * 4, sizeof(float));
                if (memblock_set(ctx, &write_out_mem_53006, &mem_53005,
                                 "mem_53005") != 0)
                    return 1;
                if (memblock_unref(ctx, &mem_53005, "mem_53005") != 0)
                    return 1;
            }
            memmove(mem_52963.mem + i_52610 * N_51954 * 4,
                    write_out_mem_53006.mem + 0, N_51954 * sizeof(float));
            if (memblock_unref(ctx, &write_out_mem_53006,
                               "write_out_mem_53006") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_53006,
                               "write_out_mem_53006") != 0)
                return 1;
            if (memblock_unref(ctx, &write_out_mem_52999,
                               "write_out_mem_52999") != 0)
                return 1;
        }
        memmove(mem_52958.mem + i_52610 * 4, mem_52974.mem + i_52232 * 4,
                sizeof(int32_t));
    }
    if (memblock_unref(ctx, &mem_52945, "mem_52945") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52974, "mem_52974") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52977, "mem_52977") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52984, "mem_52984") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52987, "mem_52987") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52992, "mem_52992") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53002, "mem_53002") != 0)
        return 1;
    
    struct memblock mem_53014;
    
    mem_53014.references = NULL;
    if (memblock_alloc(ctx, &mem_53014, bytes_52956, "mem_53014"))
        return 1;
    
    struct memblock mem_53017;
    
    mem_53017.references = NULL;
    if (memblock_alloc(ctx, &mem_53017, bytes_52956, "mem_53017"))
        return 1;
    
    struct memblock mem_53020;
    
    mem_53020.references = NULL;
    if (memblock_alloc(ctx, &mem_53020, bytes_52956, "mem_53020"))
        return 1;
    for (int32_t i_52624 = 0; i_52624 < m_51955; i_52624++) {
        int32_t res_52309;
        int32_t redout_52614 = 0;
        
        for (int32_t i_52615 = 0; i_52615 < n_51959; i_52615++) {
            float x_52313 = ((float *) images_mem_52807.mem)[i_52624 * N_51956 +
                                                             i_52615];
            bool res_52314;
            
            res_52314 = futrts_isnan32(x_52313);
            
            bool cond_52315 = !res_52314;
            int32_t res_52316;
            
            if (cond_52315) {
                res_52316 = 1;
            } else {
                res_52316 = 0;
            }
            
            int32_t res_52312 = res_52316 + redout_52614;
            int32_t redout_tmp_53119 = res_52312;
            
            redout_52614 = redout_tmp_53119;
        }
        res_52309 = redout_52614;
        
        float res_52317;
        float redout_52616 = 0.0F;
        
        for (int32_t i_52617 = 0; i_52617 < n_51959; i_52617++) {
            float y_error_elem_52322 = ((float *) mem_52963.mem)[i_52624 *
                                                                 N_51954 +
                                                                 i_52617];
            bool cond_52323 = slt32(i_52617, res_52309);
            float res_52324;
            
            if (cond_52323) {
                res_52324 = y_error_elem_52322;
            } else {
                res_52324 = 0.0F;
            }
            
            float res_52325 = res_52324 * res_52324;
            float res_52320 = res_52325 + redout_52616;
            float redout_tmp_53120 = res_52320;
            
            redout_52616 = redout_tmp_53120;
        }
        res_52317 = redout_52616;
        
        int32_t r32_arg_52326 = res_52309 - k2p2_51976;
        float res_52327 = sitofp_i32_f32(r32_arg_52326);
        float sqrt_arg_52328 = res_52317 / res_52327;
        float res_52329;
        
        res_52329 = futrts_sqrt32(sqrt_arg_52328);
        
        float res_52330 = sitofp_i32_f32(res_52309);
        float t32_arg_52331 = hfrac_51961 * res_52330;
        int32_t res_52332 = fptosi_f32_i32(t32_arg_52331);
        
        ((int32_t *) mem_53014.mem)[i_52624] = res_52332;
        ((int32_t *) mem_53017.mem)[i_52624] = res_52309;
        ((float *) mem_53020.mem)[i_52624] = res_52329;
    }
    
    int32_t res_52336;
    int32_t redout_52628 = 0;
    
    for (int32_t i_52629 = 0; i_52629 < m_51955; i_52629++) {
        int32_t x_52340 = ((int32_t *) mem_53014.mem)[i_52629];
        int32_t res_52339 = smax32(x_52340, redout_52628);
        int32_t redout_tmp_53121 = res_52339;
        
        redout_52628 = redout_tmp_53121;
    }
    res_52336 = redout_52628;
    
    struct memblock mem_53029;
    
    mem_53029.references = NULL;
    if (memblock_alloc(ctx, &mem_53029, bytes_52956, "mem_53029"))
        return 1;
    for (int32_t i_52634 = 0; i_52634 < m_51955; i_52634++) {
        int32_t x_52344 = ((int32_t *) mem_53017.mem)[i_52634];
        int32_t x_52345 = ((int32_t *) mem_53014.mem)[i_52634];
        float res_52346;
        float redout_52630 = 0.0F;
        
        for (int32_t i_52631 = 0; i_52631 < res_52336; i_52631++) {
            bool cond_52351 = slt32(i_52631, x_52345);
            float res_52352;
            
            if (cond_52351) {
                int32_t x_52353 = x_52344 + i_52631;
                int32_t x_52354 = x_52353 - x_52345;
                int32_t i_52355 = 1 + x_52354;
                float res_52356 = ((float *) mem_52963.mem)[i_52634 * N_51954 +
                                                            i_52355];
                
                res_52352 = res_52356;
            } else {
                res_52352 = 0.0F;
            }
            
            float res_52349 = res_52352 + redout_52630;
            float redout_tmp_53123 = res_52349;
            
            redout_52630 = redout_tmp_53123;
        }
        res_52346 = redout_52630;
        ((float *) mem_53029.mem)[i_52634] = res_52346;
    }
    
    int32_t iota_arg_52358 = N_51954 - n_51959;
    bool bounds_invalid_upwards_52359 = slt32(iota_arg_52358, 0);
    bool eq_x_zz_52360 = 0 == iota_arg_52358;
    bool not_p_52361 = !bounds_invalid_upwards_52359;
    bool p_and_eq_x_y_52362 = eq_x_zz_52360 && not_p_52361;
    bool dim_zzero_52363 = bounds_invalid_upwards_52359 || p_and_eq_x_y_52362;
    bool both_empty_52364 = eq_x_zz_52360 && dim_zzero_52363;
    bool eq_x_y_52365 = iota_arg_52358 == 0;
    bool p_and_eq_x_y_52366 = bounds_invalid_upwards_52359 && eq_x_y_52365;
    bool dim_match_52367 = not_p_52361 || p_and_eq_x_y_52366;
    bool empty_or_match_52368 = both_empty_52364 || dim_match_52367;
    bool empty_or_match_cert_52369;
    
    if (!empty_or_match_52368) {
        ctx->error = msgprintf("Error at %s:\n%s%s%s%d%s%s\n",
                               "bfast-irreg.fut:131:1-282:20 -> bfast-irreg.fut:240:22-31 -> /futlib/array.fut:61:1-62:12",
                               "Function return value does not match shape of type ",
                               "*", "[", iota_arg_52358, "]", "intrinsics.i32");
        if (memblock_unref(ctx, &mem_53029, "mem_53029") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53020, "mem_53020") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53017, "mem_53017") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53014, "mem_53014") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53002, "mem_53002") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52992, "mem_52992") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52987, "mem_52987") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52984, "mem_52984") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52977, "mem_52977") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52974, "mem_52974") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52968, "mem_52968") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52963, "mem_52963") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52958, "mem_52958") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52945, "mem_52945") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52930, "mem_52930") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52915, "mem_52915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52904, "mem_52904") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52898, "mem_52898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52894, "mem_52894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52860, "mem_52860") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52843, "mem_52843") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                           "lifted_1_zlzb_arg_mem_52838") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
            return 1;
        return 1;
    }
    
    int32_t x_52371 = 1 + n_51959;
    bool index_certs_52372;
    
    if (!x_52233) {
        ctx->error = msgprintf("Error at %s:\n%s%d%s%d%s\n",
                               "bfast-irreg.fut:131:1-282:20 -> bfast-irreg.fut:236:15-240:32 -> bfast-irreg.fut:238:63-81",
                               "Index [", i_52232,
                               "] out of bounds for array of shape [", N_51954,
                               "].");
        if (memblock_unref(ctx, &mem_53029, "mem_53029") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53020, "mem_53020") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53017, "mem_53017") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53014, "mem_53014") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_53002, "mem_53002") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52992, "mem_52992") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52987, "mem_52987") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52984, "mem_52984") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52977, "mem_52977") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52974, "mem_52974") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52968, "mem_52968") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52963, "mem_52963") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52958, "mem_52958") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52945, "mem_52945") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52930, "mem_52930") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52915, "mem_52915") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52904, "mem_52904") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52898, "mem_52898") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52894, "mem_52894") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52860, "mem_52860") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_52843, "mem_52843") != 0)
            return 1;
        if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                           "lifted_1_zlzb_arg_mem_52838") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
            return 1;
        if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
            return 1;
        return 1;
    }
    
    int32_t r32_arg_52373 = ((int32_t *) mappingindices_mem_52806.mem)[i_52232];
    float res_52374 = sitofp_i32_f32(r32_arg_52373);
    int64_t binop_x_53033 = sext_i32_i64(iota_arg_52358);
    int64_t bytes_53032 = 4 * binop_x_53033;
    struct memblock mem_53034;
    
    mem_53034.references = NULL;
    if (memblock_alloc(ctx, &mem_53034, bytes_53032, "mem_53034"))
        return 1;
    for (int32_t i_52638 = 0; i_52638 < iota_arg_52358; i_52638++) {
        int32_t t_52377 = x_52371 + i_52638;
        int32_t i_52378 = t_52377 - 1;
        int32_t time_52379 =
                ((int32_t *) mappingindices_mem_52806.mem)[i_52378];
        float res_52380 = sitofp_i32_f32(time_52379);
        float logplus_arg_52381 = res_52380 / res_52374;
        bool cond_52382 = 2.7182817F < logplus_arg_52381;
        float res_52383;
        
        if (cond_52382) {
            float res_52384;
            
            res_52384 = futrts_log32(logplus_arg_52381);
            res_52383 = res_52384;
        } else {
            res_52383 = 1.0F;
        }
        
        float res_52385;
        
        res_52385 = futrts_sqrt32(res_52383);
        
        float res_52386 = lam_51962 * res_52385;
        
        ((float *) mem_53034.mem)[i_52638] = res_52386;
    }
    
    struct memblock mem_53039;
    
    mem_53039.references = NULL;
    if (memblock_alloc(ctx, &mem_53039, bytes_52956, "mem_53039"))
        return 1;
    
    struct memblock mem_53042;
    
    mem_53042.references = NULL;
    if (memblock_alloc(ctx, &mem_53042, bytes_52956, "mem_53042"))
        return 1;
    
    struct memblock mem_53047;
    
    mem_53047.references = NULL;
    if (memblock_alloc(ctx, &mem_53047, bytes_53032, "mem_53047"))
        return 1;
    for (int32_t i_52654 = 0; i_52654 < m_51955; i_52654++) {
        int32_t x_52389 = ((int32_t *) mem_52958.mem)[i_52654];
        int32_t x_52390 = ((int32_t *) mem_53017.mem)[i_52654];
        float x_52391 = ((float *) mem_53020.mem)[i_52654];
        int32_t x_52392 = ((int32_t *) mem_53014.mem)[i_52654];
        float x_52393 = ((float *) mem_53029.mem)[i_52654];
        int32_t y_52396 = x_52389 - x_52390;
        float res_52397 = sitofp_i32_f32(x_52390);
        float res_52398;
        
        res_52398 = futrts_sqrt32(res_52397);
        
        float y_52399 = x_52391 * res_52398;
        float discard_52645;
        float scanacc_52641 = 0.0F;
        
        for (int32_t i_52643 = 0; i_52643 < iota_arg_52358; i_52643++) {
            bool cond_52416 = sle32(y_52396, i_52643);
            float res_52417;
            
            if (cond_52416) {
                res_52417 = 0.0F;
            } else {
                bool cond_52418 = i_52643 == 0;
                float res_52419;
                
                if (cond_52418) {
                    res_52419 = x_52393;
                } else {
                    int32_t x_52420 = x_52390 - x_52392;
                    int32_t i_52421 = x_52420 + i_52643;
                    float negate_arg_52422 = ((float *) mem_52963.mem)[i_52654 *
                                                                       N_51954 +
                                                                       i_52421];
                    float x_52423 = 0.0F - negate_arg_52422;
                    int32_t i_52424 = x_52390 + i_52643;
                    float y_52425 = ((float *) mem_52963.mem)[i_52654 *
                                                              N_51954 +
                                                              i_52424];
                    float res_52426 = x_52423 + y_52425;
                    
                    res_52419 = res_52426;
                }
                res_52417 = res_52419;
            }
            
            float res_52414 = res_52417 + scanacc_52641;
            
            ((float *) mem_53047.mem)[i_52643] = res_52414;
            
            float scanacc_tmp_53127 = res_52414;
            
            scanacc_52641 = scanacc_tmp_53127;
        }
        discard_52645 = scanacc_52641;
        
        bool acc0_52432;
        int32_t acc0_52433;
        float acc0_52434;
        bool redout_52646;
        int32_t redout_52647;
        float redout_52648;
        
        redout_52646 = 0;
        redout_52647 = -1;
        redout_52648 = 0.0F;
        for (int32_t i_52649 = 0; i_52649 < iota_arg_52358; i_52649++) {
            float x_52449 = ((float *) mem_53047.mem)[i_52649];
            float x_52450 = ((float *) mem_53034.mem)[i_52649];
            int32_t x_52451 = i_52649;
            float res_52453 = x_52449 / y_52399;
            bool cond_52454 = slt32(i_52649, y_52396);
            bool res_52455;
            
            res_52455 = futrts_isnan32(res_52453);
            
            bool res_52456 = !res_52455;
            bool x_52457 = cond_52454 && res_52456;
            float res_52458 = (float) fabs(res_52453);
            bool res_52459 = x_52450 < res_52458;
            bool x_52460 = x_52457 && res_52459;
            float res_52461;
            
            if (cond_52454) {
                res_52461 = res_52453;
            } else {
                res_52461 = 0.0F;
            }
            
            bool res_52441;
            int32_t res_52442;
            
            if (redout_52646) {
                res_52441 = redout_52646;
                res_52442 = redout_52647;
            } else {
                bool x_52444 = !x_52460;
                bool y_52445 = x_52444 && redout_52646;
                bool res_52446 = y_52445 || x_52460;
                int32_t res_52447;
                
                if (x_52460) {
                    res_52447 = x_52451;
                } else {
                    res_52447 = redout_52647;
                }
                res_52441 = res_52446;
                res_52442 = res_52447;
            }
            
            float res_52448 = res_52461 + redout_52648;
            bool redout_tmp_53129 = res_52441;
            int32_t redout_tmp_53130 = res_52442;
            float redout_tmp_53131;
            
            redout_tmp_53131 = res_52448;
            redout_52646 = redout_tmp_53129;
            redout_52647 = redout_tmp_53130;
            redout_52648 = redout_tmp_53131;
        }
        acc0_52432 = redout_52646;
        acc0_52433 = redout_52647;
        acc0_52434 = redout_52648;
        
        int32_t res_52468;
        
        if (acc0_52432) {
            res_52468 = acc0_52433;
        } else {
            res_52468 = -1;
        }
        
        bool cond_52470 = !acc0_52432;
        int32_t fst_breakzq_52471;
        
        if (cond_52470) {
            fst_breakzq_52471 = -1;
        } else {
            bool cond_52472 = slt32(res_52468, y_52396);
            int32_t res_52473;
            
            if (cond_52472) {
                int32_t i_52474 = x_52390 + res_52468;
                int32_t x_52475 = ((int32_t *) mem_52968.mem)[i_52654 *
                                                              N_51954 +
                                                              i_52474];
                int32_t res_52476 = x_52475 - n_51959;
                
                res_52473 = res_52476;
            } else {
                res_52473 = -1;
            }
            
            int32_t x_52477 = res_52473 - 1;
            int32_t x_52478 = sdiv32(x_52477, 2);
            int32_t x_52479 = 2 * x_52478;
            int32_t res_52480 = 1 + x_52479;
            
            fst_breakzq_52471 = res_52480;
        }
        
        bool cond_52481 = sle32(x_52390, 5);
        bool res_52482 = sle32(y_52396, 5);
        bool x_52483 = !cond_52481;
        bool y_52484 = res_52482 && x_52483;
        bool cond_52485 = cond_52481 || y_52484;
        int32_t fst_breakzq_52486;
        
        if (cond_52485) {
            fst_breakzq_52486 = -2;
        } else {
            fst_breakzq_52486 = fst_breakzq_52471;
        }
        ((int32_t *) mem_53039.mem)[i_52654] = fst_breakzq_52486;
        ((float *) mem_53042.mem)[i_52654] = acc0_52434;
    }
    if (memblock_unref(ctx, &mem_52958, "mem_52958") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52963, "mem_52963") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52968, "mem_52968") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53014, "mem_53014") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53017, "mem_53017") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53020, "mem_53020") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53029, "mem_53029") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53034, "mem_53034") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53047, "mem_53047") != 0)
        return 1;
    out_arrsizze_53075 = m_51955;
    out_arrsizze_53077 = m_51955;
    if (memblock_set(ctx, &out_mem_53074, &mem_53039, "mem_53039") != 0)
        return 1;
    if (memblock_set(ctx, &out_mem_53076, &mem_53042, "mem_53042") != 0)
        return 1;
    (*out_mem_p_53132).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53132, &out_mem_53074, "out_mem_53074") !=
        0)
        return 1;
    *out_out_arrsizze_53133 = out_arrsizze_53075;
    (*out_mem_p_53134).references = NULL;
    if (memblock_set(ctx, &*out_mem_p_53134, &out_mem_53076, "out_mem_53076") !=
        0)
        return 1;
    *out_out_arrsizze_53135 = out_arrsizze_53077;
    if (memblock_unref(ctx, &mem_53047, "mem_53047") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53042, "mem_53042") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53039, "mem_53039") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53034, "mem_53034") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53029, "mem_53029") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53020, "mem_53020") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53017, "mem_53017") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53014, "mem_53014") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_53002, "mem_53002") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52992, "mem_52992") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52987, "mem_52987") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52984, "mem_52984") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52977, "mem_52977") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52974, "mem_52974") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52968, "mem_52968") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52963, "mem_52963") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52958, "mem_52958") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52945, "mem_52945") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52930, "mem_52930") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52915, "mem_52915") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52904, "mem_52904") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52898, "mem_52898") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52894, "mem_52894") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52860, "mem_52860") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_52843, "mem_52843") != 0)
        return 1;
    if (memblock_unref(ctx, &lifted_1_zlzb_arg_mem_52838,
                       "lifted_1_zlzb_arg_mem_52838") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53076, "out_mem_53076") != 0)
        return 1;
    if (memblock_unref(ctx, &out_mem_53074, "out_mem_53074") != 0)
        return 1;
    return 0;
}
struct futhark_f32_1d {
    struct memblock mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + 0, dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0)
{
    struct futhark_f32_1d *bad = NULL;
    struct futhark_f32_1d *arr =
                          (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + offset, dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, arr->shape[0] * sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr)
{
    return arr->shape;
}
struct futhark_f32_2d {
    struct memblock mem;
    int64_t shape[2];
} ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int64_t dim0,
                                          int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * dim1 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + 0, dim0 * dim1 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0, int64_t dim1)
{
    struct futhark_f32_2d *bad = NULL;
    struct futhark_f32_2d *arr =
                          (struct futhark_f32_2d *) malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * dim1 * sizeof(float), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    memmove(arr->mem.mem + 0, data + offset, dim0 * dim1 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, arr->shape[0] * arr->shape[1] *
            sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_2d(struct futhark_context *ctx,
                                struct futhark_f32_2d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr)
{
    return arr->shape;
}
struct futhark_i32_1d {
    struct memblock mem;
    int64_t shape[1];
} ;
struct futhark_i32_1d *futhark_new_i32_1d(struct futhark_context *ctx,
                                          int32_t *data, int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * sizeof(int32_t), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + 0, dim0 * sizeof(int32_t));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_i32_1d *futhark_new_raw_i32_1d(struct futhark_context *ctx,
                                              char *data, int offset,
                                              int64_t dim0)
{
    struct futhark_i32_1d *bad = NULL;
    struct futhark_i32_1d *arr =
                          (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * sizeof(int32_t), "arr->mem"))
        return NULL;
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + offset, dim0 * sizeof(int32_t));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_i32_1d(struct futhark_context *ctx, struct futhark_i32_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_i32_1d(struct futhark_context *ctx,
                          struct futhark_i32_1d *arr, int32_t *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, arr->shape[0] * sizeof(int32_t));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_i32_1d(struct futhark_context *ctx,
                                struct futhark_i32_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_i32_1d(struct futhark_context *ctx,
                              struct futhark_i32_1d *arr)
{
    return arr->shape;
}
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_i32_1d **out0,
                       struct futhark_f32_1d **out1, const int32_t in0, const
                       int32_t in1, const int32_t in2, const float in3, const
                       float in4, const float in5, const
                       struct futhark_i32_1d *in6, const
                       struct futhark_f32_2d *in7)
{
    struct memblock mappingindices_mem_52806;
    
    mappingindices_mem_52806.references = NULL;
    
    struct memblock images_mem_52807;
    
    images_mem_52807.references = NULL;
    
    int32_t N_51954;
    int32_t m_51955;
    int32_t N_51956;
    int32_t trend_51957;
    int32_t k_51958;
    int32_t n_51959;
    float freq_51960;
    float hfrac_51961;
    float lam_51962;
    struct memblock out_mem_53074;
    
    out_mem_53074.references = NULL;
    
    int32_t out_arrsizze_53075;
    struct memblock out_mem_53076;
    
    out_mem_53076.references = NULL;
    
    int32_t out_arrsizze_53077;
    
    lock_lock(&ctx->lock);
    trend_51957 = in0;
    k_51958 = in1;
    n_51959 = in2;
    freq_51960 = in3;
    hfrac_51961 = in4;
    lam_51962 = in5;
    mappingindices_mem_52806 = in6->mem;
    N_51954 = in6->shape[0];
    images_mem_52807 = in7->mem;
    m_51955 = in7->shape[0];
    N_51956 = in7->shape[1];
    
    int ret = futrts_main(ctx, &out_mem_53074, &out_arrsizze_53075,
                          &out_mem_53076, &out_arrsizze_53077,
                          mappingindices_mem_52806, images_mem_52807, N_51954,
                          m_51955, N_51956, trend_51957, k_51958, n_51959,
                          freq_51960, hfrac_51961, lam_51962);
    
    if (ret == 0) {
        assert((*out0 =
                (struct futhark_i32_1d *) malloc(sizeof(struct futhark_i32_1d))) !=
            NULL);
        (*out0)->mem = out_mem_53074;
        (*out0)->shape[0] = out_arrsizze_53075;
        assert((*out1 =
                (struct futhark_f32_1d *) malloc(sizeof(struct futhark_f32_1d))) !=
            NULL);
        (*out1)->mem = out_mem_53076;
        (*out1)->shape[0] = out_arrsizze_53077;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
