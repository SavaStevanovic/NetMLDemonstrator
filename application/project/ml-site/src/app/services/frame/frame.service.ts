import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import SockJS from 'sockjs-client';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class FrameService {

  private frameSocketUrl = environment.frame_upload_stream;
  sockjs: any;
  constructor(private httpClient: HttpClient) { }

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
  }
}
