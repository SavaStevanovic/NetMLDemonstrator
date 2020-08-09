import { Injectable } from '@angular/core';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { Observable } from 'rxjs';
import { EMPTY, Subject } from 'rxjs';
import { catchError, tap, switchAll } from 'rxjs/operators';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class StreamService {

  private streamUrl = 'http://127.0.0.1:4321/frame_upload';

  constructor(private http: HttpClient) { }

  // sendFrame(data, frame) {
  //   return this.http.post<Filter[]>(this.filtersUrl);
  // }
}
