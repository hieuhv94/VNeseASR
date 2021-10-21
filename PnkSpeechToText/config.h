/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#ifndef CONFIG_H
#define CONFIG_H
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
// #include "flashlight/lib/text/dictionary/Utils.h"
// #include "flashlight/lib/text/decoder/Decoder.h"
#include "models/asr/decoder/Decoder.h"
#include "models/text_correction/Corrector.h"
using namespace w2l;
using namespace streaming;

namespace pnk
{
    // Define configuration key string
#define ACOUSTIC_MODEL_PATH_KEYSTRING "am"
#define FEATURE_EXTRACTION_PATH_KEYSTRING "feature"
#define TOKENS_PATH_KEYSTRING "tokens"
#define LEXICON_PATH_KEYSTRING "lexicon"
#define LANGUAGE_MODEL_PATH_KEYSTRING "lm"
#define LANGUAGE_MODEL_NUMBER_PATH_KEYSTRING "lm_num"
#define PUNCTUATION_ENABLE_KEYSTRING "punc_enable"
#define NER_ENABLE_KEYSTRING "ner_enable"
#define PUNCTUATION_PREDICTION_PATH_KEYSTRING "punc_model"
#define TRANSITIONS_PATH_KEYSTRING "transitions"
#define CRITERION_TYPE_KEYSTRING "criterion"
#define BEAM_SIZE_KEYSTRING "beamsize"
#define BEAM_SIZE_TOKEN_KEYSTRING "beamsizetoken"
#define BEAM_THRESHOLD_KEYSTRING "beamthreshold"
#define LANGUAGE_MODEL_WEIGHT_KEYSTRING "lmweight"
#define WORD_SCORE_KEYSTRING "wrdscore"
#define UNKNOW_SCORE_KEYSTRING "unkscore"
#define SILIENT_SCORE_KEYSTRING "silscore"
#define EOS_SCORE_KEYSTRING "eosscore"
#define LOG_ADD_KEYSTRING "logadd"
#define SMEARING_KEYSTRING "smearing"
#define INTERUPT_TIME_KEYSTRING "interupt"
#define MIN_CHUNK_SIZE_KETSTRING "minchunk"
    typedef struct AsrModulesConfig
    {
        // Path of acoustic model
        std::string amPath;
        // Path of tokens file
        std::string tokensPath;
        // Path of lexicon file
        std::string lexiconPath;
        // Feature extraction file
        std::string featurePath;
        // Path of language model
        std::string lmPath = "";
        // Path of transitions file
        std::string transPath = "";
        fl::lib::text::CriterionType criterionType = fl::lib::text::CriterionType::CTC;
        uint beamSize = 100;
        uint beamSizeToken = 100;
        double beamThreshold = 100;
        double lmWeight = 0.674;
        double wordScore = 0.628;
        double unknownScore = 0;
        double silientScore = 0;
        double eosScore = 0;
        bool logAdd = false;
        fl::lib::text::SmearingMode smearing =  fl::lib::text::SmearingMode::NONE;
 
    } AsrModulesConfig;
    typedef struct TextCorrectionModulesConfig
    {
        bool punc = false;
        bool ner = false;
        // Path of language model for word to num
        std::string lmNumPath = "";
        // Path of prediction punctuation model
        std::string puncPath = "";
    } TextCorrectionModulesConfig;

    typedef struct SessionConfig
    {
       int interuptTime = 1000; //msec
        int minChunkMsec = 200; //msec
        int tokensSize;
    } SessionConfig;

    typedef struct ServiceConfig
    {
        AsrModulesConfig asr;
        TextCorrectionModulesConfig textCorrection;
        SessionConfig session;
    } ServiceConfig;
    
    inline bool checkFileExisted(std::string filePath)
    {
        struct stat buffer;
        return (stat(filePath.c_str(), &buffer) == 0);
    }
    inline std::string &ltrim(std::string &s)
    {
        auto it = std::find_if(s.begin(), s.end(),
                               [](char c) {
                                   return !std::isspace<char>(c, std::locale::classic());
                               });
        s.erase(s.begin(), it);
        return s;
    }

    inline std::string &rtrim(std::string &s)
    {
        auto it = std::find_if(s.rbegin(), s.rend(),
                               [](char c) {
                                   return !std::isspace<char>(c, std::locale::classic());
                               });
        s.erase(it.base(), s.end());
        return s;
    }

    inline std::string &trim(std::string &s)
    {
        return ltrim(rtrim(s));
    }
    static void parserConfigFile(const std::string configPath, ServiceConfig& config) {
        if (!checkFileExisted(configPath))
        {
            throw std::runtime_error(std::string("[Configuration] Cannot found config file at ") + configPath);
            return;
        }
        uint lineCount = 1;
        std::ifstream confStringStream(configPath.c_str());
        std::string line;
        while (std::getline(confStringStream, line))
        {
            line = line.substr(0, line.find("#"));
            if (line.empty())
                continue;
            if (line.find('=') == std::string::npos)
            {
                throw std::runtime_error(std::string("[Configuration] Config file is wrong format at line ") + std::to_string(lineCount));
                return;
            }
            std::istringstream obj(line);
            std::string key;
            if (std::getline(obj, key, '='))
            {
                std::string value;
                if (std::getline(obj, value))
                {
                    key = trim(key);
                    value = trim(value);
                    if (!key.compare(ACOUSTIC_MODEL_PATH_KEYSTRING))
                        config.asr.amPath = value;
                    else if (!key.compare(FEATURE_EXTRACTION_PATH_KEYSTRING))
                        config.asr.featurePath = value;
                    else if (!key.compare(TOKENS_PATH_KEYSTRING))
                        config.asr.tokensPath = value;
                    else if (!key.compare(LEXICON_PATH_KEYSTRING))
                        config.asr.lexiconPath = value;
                    else if (!key.compare(PUNCTUATION_ENABLE_KEYSTRING))
                    {
                        if (value.compare("true") == 0)
                            config.textCorrection.punc = true;
                        else if (value.compare("false") == 0)
                            config.textCorrection.punc = false;
                        else
                        {
                            throw std::runtime_error(std::string("[Configuration] Parsing punc_enable parameter is failed"));
                            return;
                        }
                    }
                    else if (!key.compare(NER_ENABLE_KEYSTRING))
                    {
                        if (value.compare("true") == 0)
                            config.textCorrection.ner = true;
                        else if (value.compare("false") == 0)
                            config.textCorrection.ner = false;
                        else
                        {
                            throw std::runtime_error(std::string("[Configuration] Parsing ner_enable parameter is failed"));
                            return;
                        }
                    }
                    else if (!key.compare(LANGUAGE_MODEL_NUMBER_PATH_KEYSTRING))
                        config.textCorrection.lmNumPath = value;
                        else if (!key.compare(PUNCTUATION_PREDICTION_PATH_KEYSTRING))
                        config.textCorrection.puncPath = value;
                    else if (!key.compare(LANGUAGE_MODEL_PATH_KEYSTRING))
                        config.asr.lmPath = value;
                    else if (!key.compare(TRANSITIONS_PATH_KEYSTRING))
                        config.asr.transPath = value;
                    else if (!key.compare(CRITERION_TYPE_KEYSTRING))
                    {
                        if (!value.compare("ctc"))
                        {
                            config.asr.criterionType = fl::lib::text::CriterionType::CTC;
                        }
                        else if (!value.compare("asg"))
                        {
                            config.asr.criterionType = fl::lib::text::CriterionType::ASG;
                        }
                        else if (!value.compare("s2s"))
                        {
                            config.asr.criterionType = fl::lib::text::CriterionType::S2S;
                        }
                        else
                        {
                            LOG(WARNING) << "Criterion type " << value << " is not supported, use default value is ctc";
                        }
                    }
                    else if (!key.compare(BEAM_SIZE_KEYSTRING))
                        config.asr.beamSize = std::stoi(value);
                    else if (!key.compare(BEAM_SIZE_TOKEN_KEYSTRING))
                        config.asr.beamSizeToken = std::stoi(value);
                    else if (!key.compare(BEAM_THRESHOLD_KEYSTRING))
                        config.asr.beamThreshold = std::stod(value);
                    else if (!key.compare(LANGUAGE_MODEL_WEIGHT_KEYSTRING))
                        config.asr.lmWeight = std::stod(value);
                    else if (!key.compare(WORD_SCORE_KEYSTRING))
                        config.asr.wordScore = std::stod(value);
                    else if (!key.compare(UNKNOW_SCORE_KEYSTRING))
                        config.asr.unknownScore = std::stod(value);
                    else if (!key.compare(SILIENT_SCORE_KEYSTRING))
                        config.asr.silientScore = std::stod(value);
                    else if (!key.compare(EOS_SCORE_KEYSTRING))
                        config.asr.eosScore = std::stod(value);
                    else if (!key.compare(LOG_ADD_KEYSTRING))
                    {
                        if (value.compare("true") == 0)
                            config.asr.logAdd = true;
                        else if (value.compare("false") == 0)
                            config.asr.logAdd = false;
                        else
                        {
                            throw std::runtime_error(std::string("[Configuration] Parsing log add parameter is failed"));
                            return;
                        }
                    }
                    else if (!key.compare(SMEARING_KEYSTRING))
                    {
                        if (!value.compare("none"))
                        {
                            config.asr.smearing =  fl::lib::text::SmearingMode::NONE;
                        }
                        else if (!value.compare("max"))
                        {
                            config.asr.smearing =  fl::lib::text::SmearingMode::MAX;
                        }
                        else if (!value.compare("logadd"))
                        {
                            config.asr.smearing =  fl::lib::text::SmearingMode::LOGADD;
                        }
                        else
                        {
                            LOG(WARNING) << "Smearing mode " << value << " is not supported, use default value is none";
                        }
                    }
                    else if (!key.compare(INTERUPT_TIME_KEYSTRING))
                        config.session.interuptTime = std::stod(value);
                    else if (!key.compare(MIN_CHUNK_SIZE_KETSTRING))
                        config.session.minChunkMsec = std::stod(value);
                    else
                    {
                        throw std::runtime_error(std::string("[Configuration] Invalid config parameter ") + key);
                        return;
                    }
                }
            }
            lineCount++;
        }
    }
}
#endif //CONFIG_H