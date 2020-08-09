import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map, catchError } from 'rxjs/operators';
import { from, Observable, throwError } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class FrameService {

  private frameUrl = 'http://127.0.0.1:4321/frame_upload';

  constructor(private httpClient: HttpClient) { }

  processFrame(frame, config) {
    var data = {'frame': frame, 'config': config}
    return this.httpClient.post(this.frameUrl, data).
        pipe(
           map((data: any) => {
             return data;
           }), catchError( error => {
             return throwError( 'Something went wrong!' );
           })
        ).subscribe()
    }
}
