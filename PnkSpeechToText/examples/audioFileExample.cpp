/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "PnkS2t.h"
#include <iostream>
#include <time.h>
void help() {
    std::cout << "Usage: \n" <<
    "Sync mode:         ./audioFileExample config_file.cfg audio_file.wav\n" <<
    "Interactive mode:  ./audioFileExample config_file.cfg\n";
}
int main(int argc, char* argv[]) {
    if ((argc < 2) || (argc > 3))
    {
        std::cout << "Invalid arguments\n";
        help();
        return 0;
    }
    pnk::PnkSpeechToText* manager = new pnk::PnkSpeechToText(argv[1]);
	pnk::FilePlugin* filePlugin = manager->getFilePlugin();
    if (argc == 3)
    {
        std::string output = "";
        clock_t tStart = clock();
        filePlugin->runSync(argv[2], output);
        printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
        std::cout << "Text: " << output << std::endl;
    } else {
        filePlugin->runInteractiveMode();
    }
}