import { Injectable, OnInit } from '@angular/core';
import SockJS from 'sockjs-client';
import { StateService } from '../state/state.service';

@Injectable({
  providedIn: 'root'
})
export class FrameService{
  socket: SockJS;

  constructor(
    private stateService: StateService) {
    }

  private isConnectionOpen(): boolean {
    return this.socket?.readyState == SockJS.OPEN;
  }

  public openImageConnection(frameSocketUrl): SockJS {
    if (this.socket)
      this.socket.close();

    if (!this.isConnectionOpen()){
      this.socket = new SockJS(frameSocketUrl);
      this.stateService.videoStart$.subscribe(play => this.closeConnection(play))
      this.socket.onclose = () => this.stateService.setVideoPlaying(false)
      this.socket.onerror = () => this.stateService.setVideoPlaying(false)
    }
    return this.socket;
  }

  private closeConnection(play): void {
    if (!play) {
      this.socket.close()
    }
  }
}
