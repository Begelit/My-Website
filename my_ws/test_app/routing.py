from django.urls import path

from .consumers import WSConsumer1, WSConsumerTransformer
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
#from test_app.routing import ws_urlpatterns

ws_urlpatterns = [
    path('ws/some_url/',WSConsumer1.as_asgi()),
    path("ws/transformerSocket_url/",WSConsumerTransformer.as_asgi()),
]

application = ProtocolTypeRouter({
    #'http': get_asgi_application(),
    'websocket': AuthMiddlewareStack(URLRouter(ws_urlpatterns))
    })