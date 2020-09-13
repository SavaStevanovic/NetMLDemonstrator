import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map, catchError } from 'rxjs/operators';
import { Observer, Observable, throwError, Subject } from 'rxjs';
import SockJS from 'sockjs-client';

@Injectable({
  providedIn: 'root'
})
export class FrameService {

  private frameUrl = 'http://127.0.0.1:4321/frame_upload';
  private frameSocketUrl = 'http://127.0.0.1:4321/frame_upload_stream';
  sockjs: any;
  constructor(private httpClient: HttpClient) { }

  processFrame(frame, config) {
    var post_data = {'frame': frame, 'config': config}
    return this.httpClient.post(this.frameUrl, post_data).pipe(
      map((data: any) => {
        return data;
      }), catchError( error => {
        return throwError( 'Something went wrong!' );
      })
    )
  }

  isConnectionOpen(){
    if (this.sockjs && this.sockjs.readyState == SockJS.OPEN)
      return true

    return false
  }

  closeConnection(){
    console.log("close")
    this.sockjs.close()
  }



  public openImageConnection() {
    if (this.isConnectionOpen())
      this.closeConnection()

    if (!this.isConnectionOpen()){
      this.sockjs = new SockJS(this.frameSocketUrl);
    }
    return this.sockjs;
    // let observable = new Observable((obs:Observer<MessageEvent>) => {
    //   wsjs.onmessage = obs.next.bind(obs)
    //   wsjs.onerror = obs.error.bind(obs)
    //   wsjs.onclose = obs.complete.bind(obs)
    //   wsjs.onopen = () =>{
    //     console.log('open')
    //     // let sendObject = {
    //     //   path: this.setImage,
    //     //   slika: imageEncoded
    //     // }

    //     // wsjs.send(JSON.stringify(sendObject));
    //     let post_data = {'frame': frame, 'config': config}
    //     wsjs.send(JSON.stringify(post_data))
    //   }
    // })

    // let subject = new Subject()
    // observable.subscribe(subject)

    // return subject
  }
}
