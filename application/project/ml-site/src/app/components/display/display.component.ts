import { Component, AfterViewInit, OnInit, ElementRef } from '@angular/core';
import { ViewChild } from '@angular/core';
import { StreamService } from '../../services/stream/stream.service';
import { catchError, tap, map } from 'rxjs/operators';

@Component({
  selector: 'app-display',
  templateUrl: './display.component.html',
  styleUrls: ['./display.component.css']
})
export class DisplayComponent implements AfterViewInit, OnInit {
  @ViewChild('videoElement') videoElement: ElementRef;
  @ViewChild('unprocessedCanvas') unprocessedCanvas: ElementRef;

  isPlaying = false;

  displayControls = true;

  constructor(private service: StreamService) {
  }

  capture() {
    this.unprocessedCanvas.nativeElement.width = this.videoElement.nativeElement.clientWidth;
    this.unprocessedCanvas.nativeElement.height = this.videoElement.nativeElement.clientHeight;
    var context = this.unprocessedCanvas.nativeElement.getContext("2d").drawImage(this.videoElement.nativeElement, 0, 0, this.videoElement.nativeElement.videoWidth, this.videoElement.nativeElement.videoHeight, 0, 0, this.videoElement.nativeElement.clientWidth, this.videoElement.nativeElement.clientHeight);
  }

  ngOnInit(): void {
  }

  ngAfterViewInit(): void {
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
          this.videoElement.nativeElement.srcObject = stream;
          // this.videoElement.nativeElement.addEventListener('timeupdate', this.capture, false);
          // this.videoElement.nativeElement.play();
      });
    }
  }

  start() {
    this.initCamera({ video: true, audio: false });
  }

  sound() {
    this.initCamera({ video: true, audio: true });
  }

  pause() {
    this.videoElement.nativeElement.pause();
  }

  toggleControls() {
    this.videoElement.nativeElement.controls = this.displayControls;
    this.displayControls = !this.displayControls;
  }

  resume() {
    this.videoElement.nativeElement.play();
  }

  initCamera(config:any) {
    var browser = <any>navigator;

    browser.getUserMedia = (browser.getUserMedia ||
      browser.webkitGetUserMedia ||
      browser.mozGetUserMedia ||
      browser.msGetUserMedia);

    browser.mediaDevices.getUserMedia(config).then(stream => {
      this.videoElement.nativeElement.srcObject = stream;
      this.videoElement.nativeElement.play();
    });
  }
}
