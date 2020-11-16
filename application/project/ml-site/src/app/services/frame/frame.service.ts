import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import SockJS from 'sockjs-client';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class FrameService {

  private frameSocketUrl = environment.frame_upload_stream;
  socket: SockJS;

  constructor(private httpClient: HttpClient) { }

  private isConnectionOpen(): boolean {
    return this.socket?.readyState == SockJS.OPEN;
  }

  public openImageConnection(): SockJS {
    if (this.isConnectionOpen())
      this.socket.close();

    if (!this.isConnectionOpen()){
      this.socket = new SockJS(this.frameSocketUrl);
    }
    return this.socket;
  }
}
