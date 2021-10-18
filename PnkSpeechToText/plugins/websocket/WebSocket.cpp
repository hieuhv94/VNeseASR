/**
 * Author: hieuhv
 * Phenikaa Smart Solution
 */
#include "WebSocketPlugin.h"
#include <algorithm>
#include <functional>

//The name of the special JSON field that holds the message type for messages
#define MESSAGE_FIELD "__MESSAGE__"

WebsocketServer::WebsocketServer(std::string certFile, std::string keyFile) : _certFile(certFile), _keyFile(keyFile)
{
    //Wire up our event handlers
    this->endpoint.set_open_handler(std::bind(&WebsocketServer::onOpen, this, std::placeholders::_1));
    this->endpoint.set_close_handler(std::bind(&WebsocketServer::onClose, this, std::placeholders::_1));
    this->endpoint.set_message_handler(std::bind(&WebsocketServer::onMessage, this, std::placeholders::_1, std::placeholders::_2));
    this->endpoint.set_tls_init_handler(std::bind(&WebsocketServer::onTtlsInit, this, MOZILLA_INTERMEDIATE, std::placeholders::_1));
    //Initialise the Asio library, using our own event loop object
    this->endpoint.init_asio(&(this->eventLoop));
}

void WebsocketServer::run(int port)
{
    //Listen on the specified port number and start accepting connections
    this->endpoint.listen(asio::ip::address_v4::any(), port);
    this->endpoint.start_accept();

    //Start the Asio event loop
    this->endpoint.run();
}

size_t WebsocketServer::numConnections()
{
    //Prevent concurrent access to the list of open connections from multiple threads
    std::lock_guard<std::mutex> lock(this->connectionListMutex);

    return this->openConnections.size();
}

void WebsocketServer::sendMessage(ClientConnection conn, const std::string message)
{
    //Send the JSON data to the client (will happen on the networking thread's event loop)
    this->endpoint.send(conn, message, websocketpp::frame::opcode::text);
}

void WebsocketServer::broadcastMessage(const std::string message)
{
    //Prevent concurrent access to the list of open connections from multiple threads
    std::lock_guard<std::mutex> lock(this->connectionListMutex);

    for (auto conn : this->openConnections)
    {
        this->sendMessage(conn, message);
    }
}

void WebsocketServer::onOpen(ClientConnection conn)
{
    {
        //Prevent concurrent access to the list of open connections from multiple threads
        std::lock_guard<std::mutex> lock(this->connectionListMutex);

        //Add the connection handle to our list of open connections
        this->openConnections.push_back(conn);
    }

    //Invoke any registered handlers
    for (auto handler : this->connectHandlers)
    {
        handler(conn);
    }
}

void WebsocketServer::onClose(ClientConnection conn)
{
    {
        //Prevent concurrent access to the list of open connections from multiple threads
        std::lock_guard<std::mutex> lock(this->connectionListMutex);

        //Remove the connection handle from our list of open connections
        auto connVal = conn.lock();
        auto newEnd = std::remove_if(this->openConnections.begin(), this->openConnections.end(), [&connVal](ClientConnection elem) {
            //If the pointer has expired, remove it from the vector
            if (elem.expired() == true)
            {
                return true;
            }

            //If the pointer is still valid, compare it to the handle for the closed connection
            auto elemVal = elem.lock();
            if (elemVal.get() == connVal.get())
            {
                return true;
            }

            return false;
        });

        //Truncate the connections vector to erase the removed elements
        this->openConnections.resize(std::distance(openConnections.begin(), newEnd));
    }

    //Invoke any registered handlers
    for (auto handler : this->disconnectHandlers)
    {
        handler(conn);
    }
}

void WebsocketServer::onMessage(ClientConnection conn, WebsocketEndpoint::message_ptr msg)
{
    for (auto handler : this->messageHandlers)
    {
        handler(conn, msg);
    }
}

std::string get_password()
{
    return "phenikaa";
}

context_ptr WebsocketServer::onTtlsInit(tls_mode mode, websocketpp::connection_hdl hdl)
{
    namespace asio = websocketpp::lib::asio;

    LOG(INFO) << "on_tls_init called with hdl: " << hdl.lock().get();
    LOG(INFO) << "using TLS mode: " << (mode == MOZILLA_MODERN ? "Mozilla Modern" : "Mozilla Intermediate");

    context_ptr ctx = websocketpp::lib::make_shared<asio::ssl::context>(asio::ssl::context::sslv23);

    try
    {
        if (mode == MOZILLA_MODERN)
        {
            // Modern disables TLSv1
            ctx->set_options(asio::ssl::context::default_workarounds |
                             asio::ssl::context::no_sslv2 |
                             asio::ssl::context::no_sslv3 |
                             asio::ssl::context::no_tlsv1 |
                             asio::ssl::context::single_dh_use);
        }
        else
        {
            ctx->set_options(asio::ssl::context::default_workarounds |
                             asio::ssl::context::no_sslv2 |
                             asio::ssl::context::no_sslv3 |
                             asio::ssl::context::single_dh_use);
        }
        ctx->set_password_callback(bind(&get_password));
        ctx->use_certificate_chain_file(_certFile);
        ctx->use_private_key_file(_keyFile, asio::ssl::context::pem);

        // Example method of generating this file:
        // `openssl dhparam -out dh.pem 2048`
        // Mozilla Intermediate suggests 1024 as the minimum size to use
        // Mozilla Modern suggests 2048 as the minimum size to use.

        std::string ciphers;

        if (mode == MOZILLA_MODERN)
        {
            ciphers = "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-DSS-AES128-GCM-SHA256:kEDH+AESGCM:ECDHE-RSA-AES128-SHA256:ECDHE-ECDSA-AES128-SHA256:ECDHE-RSA-AES128-SHA:ECDHE-ECDSA-AES128-SHA:ECDHE-RSA-AES256-SHA384:ECDHE-ECDSA-AES256-SHA384:ECDHE-RSA-AES256-SHA:ECDHE-ECDSA-AES256-SHA:DHE-RSA-AES128-SHA256:DHE-RSA-AES128-SHA:DHE-DSS-AES128-SHA256:DHE-RSA-AES256-SHA256:DHE-DSS-AES256-SHA:DHE-RSA-AES256-SHA:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!3DES:!MD5:!PSK";
        }
        else
        {
            ciphers = "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:DHE-RSA-AES128-GCM-SHA256:DHE-DSS-AES128-GCM-SHA256:kEDH+AESGCM:ECDHE-RSA-AES128-SHA256:ECDHE-ECDSA-AES128-SHA256:ECDHE-RSA-AES128-SHA:ECDHE-ECDSA-AES128-SHA:ECDHE-RSA-AES256-SHA384:ECDHE-ECDSA-AES256-SHA384:ECDHE-RSA-AES256-SHA:ECDHE-ECDSA-AES256-SHA:DHE-RSA-AES128-SHA256:DHE-RSA-AES128-SHA:DHE-DSS-AES128-SHA256:DHE-RSA-AES256-SHA256:DHE-DSS-AES256-SHA:DHE-RSA-AES256-SHA:AES128-GCM-SHA256:AES256-GCM-SHA384:AES128-SHA256:AES256-SHA256:AES128-SHA:AES256-SHA:AES:CAMELLIA:DES-CBC3-SHA:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!aECDH:!EDH-DSS-DES-CBC3-SHA:!EDH-RSA-DES-CBC3-SHA:!KRB5-DES-CBC3-SHA";
        }

        if (SSL_CTX_set_cipher_list(ctx->native_handle(), ciphers.c_str()) != 1)
        {
            LOG(ERROR) << "Error setting cipher list";
        }
    }
    catch (std::exception &e)
    {
       LOG(ERROR) << "Exception: " << e.what();
    }
    return ctx;
}