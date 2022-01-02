import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import SockJS from 'sockjs-client';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class FrameService {
  socket: SockJS;

  constructor() { }

  private isConnectionOpen(): boolean {
    return this.socket?.readyState == SockJS.OPEN;
  }

  public openImageConnection(frameSocketUrl): SockJS {
    if (this.isConnectionOpen())
      this.socket.close();

    if (!this.isConnectionOpen()){
      this.socket = new SockJS(frameSocketUrl);
    }
    return this.socket;
  }
}
