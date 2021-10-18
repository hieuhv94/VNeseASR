/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
/* Use the newer ALSA API */
#define ALSA_PCM_NEW_HW_PARAMS_API

#include <alsa/asoundlib.h>
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavutil/opt.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#ifdef __cplusplus
}
#endif
namespace pnk
{
    class ReSampler
    {
    public:
        ReSampler(int inChannelCount,
                  int outChannelCount,
                  int inChannelLayout,
                  int outChannelLayout,
                  int inSampleRate,
                  int outSampleRate,
                  AVSampleFormat inSampleFmt,
                  AVSampleFormat outSampleFmt)
        {
            _swr = swr_alloc();
            av_opt_set_int(_swr, "in_channel_count", inChannelCount, 0);
            av_opt_set_int(_swr, "out_channel_count", outChannelCount, 0);
            av_opt_set_int(_swr, "in_channel_layout", inChannelLayout, 0);
            av_opt_set_int(_swr, "out_channel_layout", outChannelLayout, 0);
            av_opt_set_int(_swr, "in_sample_rate", inSampleRate, 0);
            av_opt_set_int(_swr, "out_sample_rate", outSampleRate, 0);
            av_opt_set_sample_fmt(_swr, "in_sample_fmt", inSampleFmt, 0);
            av_opt_set_sample_fmt(_swr, "out_sample_fmt", outSampleFmt, 0);
            swr_init(_swr);
            if (!swr_is_initialized(_swr))
            {
                fprintf(stderr, "Resampler has not been properly initialized\n");
                return;
            }
        }
        ~ReSampler() {
            swr_free(&_swr);
        }
        int run(uint8_t** dst, int outCount, const uint8_t** src, int inCount)
        {
            return swr_convert(_swr, dst, outCount, src, inCount);
        }
    private:
        struct SwrContext* _swr;
    };
} // namespace pnk