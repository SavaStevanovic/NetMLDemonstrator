import { Injectable } from '@angular/core';
import { Subject, BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class StateService {

  private modelTypesSource = new Subject<string[]>();
  private videoStartSource = new BehaviorSubject<boolean>(false);
  private videoPlayingSource = new BehaviorSubject<boolean>(false);
  private videoQualitySource = new BehaviorSubject<number>(0.5);

  constructor () {
    this.videoQualitySource.next(0.5)
  }

  // Observable string streams
  modelTypes$ = this.modelTypesSource.asObservable();
  videoPlaying$ = this.videoPlayingSource.asObservable();
  videoStart$ = this.videoStartSource.asObservable();
  videoQuality$ = this.videoQualitySource.asObservable();

  // // Service message commands
  setModelTypes(modelTypes: string[]) {
    this.modelTypesSource.next(modelTypes);
  }

  setVideoPlaying(videoPlaying: boolean) {
    this.videoPlayingSource.next(videoPlaying);
  }

  setVideoStart(videoStart: boolean) {
    this.videoStartSource.next(videoStart);
  }

  setVideoQuality(videoQuality: number) {
    this.videoQualitySource.next(videoQuality);
  }
}
