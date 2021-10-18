/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#ifndef _WEBSOCKET_SERVER
#define _WEBSOCKET_SERVER

//We need to define this when using the Asio library without Boost
#define ASIO_STANDALONE

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/config/asio.hpp>
#include <websocketpp/server.hpp>

#include <functional>
#include <string>
#include <vector>
#include <mutex>
#include <map>

using std::string;
using std::vector;
using std::map;
enum tls_mode {
    MOZILLA_INTERMEDIATE = 1,
    MOZILLA_MODERN = 2
};
typedef websocketpp::lib::shared_ptr<websocketpp::lib::asio::ssl::context> context_ptr;
typedef websocketpp::server<websocketpp::config::asio_tls> WebsocketEndpoint;
typedef websocketpp::connection_hdl ClientConnection;

class WebsocketServer
{
	public:
		
		WebsocketServer(std::string certFile, std::string keyFile);
		void run(int port);
		
		//Returns the number of currently connected clients
		size_t numConnections();
		
		//Registers a callback for when a client connects
		template <typename CallbackTy>
		void connect(CallbackTy handler)
		{
			//Make sure we only access the handlers list from the networking thread
			this->eventLoop.post([this, handler]() {
				this->connectHandlers.push_back(handler);
			});
		}
		
		//Registers a callback for when a client disconnects
		template <typename CallbackTy>
		void disconnect(CallbackTy handler)
		{
			//Make sure we only access the handlers list from the networking thread
			this->eventLoop.post([this, handler]() {
				this->disconnectHandlers.push_back(handler);
			});
		}
		
		//Registers a callback for when a particular type of message is received
		template <typename CallbackTy>
		void message( CallbackTy handler)
		{
			// Generate an interrupt
			// std::raise(SIGINT);
			//Make sure we only access the handlers list from the networking thread
			this->eventLoop.post([this, handler]() {
				this->messageHandlers.push_back(handler);
			});
		}
		
		//Sends a message to an individual client
		//(Note: the data transmission will take place on the thread that called WebsocketServer::run())
		void sendMessage(ClientConnection conn, const std::string message);
		
		//Sends a message to all connected clients
		//(Note: the data transmission will take place on the thread that called WebsocketServer::run())
		void broadcastMessage(const std::string message);
		
	private:
		
		void onOpen(ClientConnection conn);
		void onClose(ClientConnection conn);
		void onMessage(ClientConnection conn, WebsocketEndpoint::message_ptr msg);
		context_ptr onTtlsInit(tls_mode mode, websocketpp::connection_hdl hdl);
		asio::io_service eventLoop;
		WebsocketEndpoint endpoint;
		vector<ClientConnection> openConnections;
		std::mutex connectionListMutex;
		
		vector<std::function<void(ClientConnection)>> connectHandlers;
		vector<std::function<void(ClientConnection)>> disconnectHandlers;
		vector<std::function<void(ClientConnection, WebsocketEndpoint::message_ptr)>> messageHandlers;
		std::string _certFile;
		std::string _keyFile;
};
#endif //_WEBSOCKET_SERVER