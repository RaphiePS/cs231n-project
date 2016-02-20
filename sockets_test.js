var server = require('http').createServer();
var io = require('socket.io')(server);
io.on('connection', function(socket){
  socket.on('image', function(data) {
  	console.log("GOT DATA!");
  	socket.emit("ack", {
        "keyLeft": false,
        "keyRight": false,
        "keyFaster": true,
        "keySlower": false
    });
  });
});
server.listen(3000);