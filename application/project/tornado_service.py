"""
    Simple sockjs-tornado chat application. By default will listen on port 8080.
"""
import tornado.ioloop
import tornado.web

import sockjs.tornado

# class IndexHandler(tornado.web.RequestHandler):
#     """Regular HTTP handler to serve the chatroom page"""
#     def get(self):
#         self.render('/')

class MessageConnection(sockjs.tornado.SockJSConnection):
    """Chat connection implementation"""
    # Class level variable
    # participants = set()
    def __init__(self, session):
        """Connection constructor.

        `session`
            Associated session
        """
        self.session = session
    def on_open(self, info):
        # Send that someone joined
        # self.broadcast(self.participants, "Someone joined.")

        # Add client to the clients list
        # self.participants.add(self)
        print('Stigao je info: {}'.format(info))
        # self.send(info)

    def on_message(self, message):
        # Broadcast message
        self.send(message)

    def on_close(self):
        # Remove client from the clients list and broadcast leave message
        # self.participants.remove(self)

        # self.broadcast(self.participants, "Someone left.")
        pass

if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.DEBUG)

    # 1. Create chat router
    ChatRouter = sockjs.tornado.SockJSRouter(MessageConnection, '/echo')

    # 2. Create Tornado application
    app = tornado.web.Application(ChatRouter.urls)

    # 3. Make Tornado app listen on port 8080
    app.listen(4320)
    
    # 4. Start IOLoop
    tornado.ioloop.IOLoop.instance().start()