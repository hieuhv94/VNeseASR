/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "PnkS2t.h"
#include <iostream>
void help() {
    std::cout << "Usage: ./websocketExample config_file.cfg port cert key\n";
}
int main(int argc, char* argv[]) {
    if ((argc != 5))
    {
        std::cout << "Invalid arguments\n";
        help();
        return 0;
    }
    pnk::PnkSpeechToText* manager = new pnk::PnkSpeechToText(argv[1]);
	pnk::WebSocketPlugin* wsPlugin = manager->getWebSocketPlugin();
    wsPlugin->run(std::stoi(argv[2]), argv[3], argv[4]);
}