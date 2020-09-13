import { Injectable } from '@angular/core';
import SockJS from 'sockjs-client';
import { Observable, Observer, Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SockjsMessageService {

  sockjs: any;
  wsuri = "http://127.0.0.1:4321/echo"

  constructor() { }

  isConnectionOpen(){
    if (this.sockjs && this.sockjs.readyState == SockJS.OPEN)
      return true

    return false
  }

  closeConnection(){
    console.log("close")
    this.sockjs.close()
  }

  public openImageConnection(imageEncoded) {
    if (this.isConnectionOpen())
      this.closeConnection()

    let wsjs = new SockJS(this.wsuri)
    this.sockjs = wsjs

    let observable = new Observable((obs:Observer<MessageEvent>) => {
      wsjs.onmessage = obs.next.bind(obs)
      wsjs.onerror = obs.error.bind(obs)
      wsjs.onclose = obs.complete.bind(obs)
      wsjs.onopen = () =>{
          console.log('open')
          // let sendObject = {
          //   path: this.setImage,
          //   slika: imageEncoded
          // }

          // wsjs.send(JSON.stringify(sendObject));
          wsjs.send('Hello world')
      }
    })

    let subject= new Subject()
    observable.subscribe(subject)

    return subject
  }

  public sendImage(imageEncoded) {
    let sendObject = {
      // path: this.setImage,
      slika: imageEncoded
    }

    this.sockjs.send(JSON.stringify(sendObject))
  }
}
