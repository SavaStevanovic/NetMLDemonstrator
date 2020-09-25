import { Component, AfterViewInit, OnInit, ElementRef } from '@angular/core';
import { ViewChild } from '@angular/core';
import { FilterService } from '../../services/filter/filter.service';
import { FrameService } from '../../services/frame/frame.service';
import { Filter } from '../../models/filter';
import { SockjsMessageService } from 'src/app/services/sockjs-message.service';
import { Observer, Observable, throwError, Subject } from 'rxjs';

@Component({
  selector: 'app-display',
  templateUrl: './display.component.html',
  styleUrls: ['./display.component.css']
})
export class DisplayComponent implements AfterViewInit, OnInit {
  @ViewChild('videoElement') videoElement: ElementRef;
  unprocessedCanvas = document.createElement('canvas');
  @ViewChild('processedCanvas') processedCanvas: ElementRef;

  filters: Filter[];
  image: any;
  processed_context: any;
  unprocessed_context: any;
  video_native_element: any;
  videoPlaying = false;
  sock: any;

  constructor(
    public frameService: FrameService,
    private filterService: FilterService) {
  }

  capture() {
    if (this.video_native_element.paused) {
      return;
    }

    this.unprocessed_context.drawImage(this.video_native_element, 0, 0);
    let post_data = {'frame': this.unprocessedCanvas.toDataURL(), 'config': this.filters}
    this.sock.send(JSON.stringify(post_data))
  }

  ngOnInit(): void {
    this.getFilters();
    this.image = new Image();
    this.image.onload = ()=> {
      this.processed_context.drawImage(this.image, 0, 0);
    }
    this.sock = this.frameService.openImageConnection()
    this.sock.onmessage = (v) =>  {
      this.image.src = JSON.parse(v.data)["image"];
      requestAnimationFrame(this.capture.bind(this))
    }
    this.initCamera({ video: true, audio: true });
  }

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters);
  }

  ngAfterViewInit(): void {
    this.processed_context = this.processedCanvas.nativeElement.getContext("2d");
    this.unprocessed_context = this.unprocessedCanvas.getContext("2d");
    this.video_native_element = this.videoElement.nativeElement;
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
          this.video_native_element.srcObject = stream;
      });
    }
  }

  pause() {
    this.video_native_element.pause();
    this.videoPlaying = this.video_native_element.paused==false;
  }

  resume() {
    this.video_native_element.play();
    this.capture()
    this.videoPlaying = this.video_native_element.paused==false;
  }

  isPlaying() {
    return this.video_native_element.paused==false;
  }
  initCamera(config:any) {
    var browser = <any>navigator;

    browser.getUserMedia = (browser.getUserMedia ||
      browser.webkitGetUserMedia ||
      browser.mozGetUserMedia ||
      browser.msGetUserMedia);

    browser.mediaDevices.getUserMedia(config).then(stream => {
      this.video_native_element.srcObject = stream;
    });
  }
  updateCanvas(){
    let nativVideoEelement = this.video_native_element
    if (this.unprocessedCanvas.width!=nativVideoEelement.clientWidth)
      this.unprocessedCanvas.width = nativVideoEelement.clientWidth;
    if (this.unprocessedCanvas.height!=nativVideoEelement.clientHeight)
      this.unprocessedCanvas.height = nativVideoEelement.clientHeight;
    if (this.processedCanvas.nativeElement.width!=nativVideoEelement.clientWidth)
      this.processedCanvas.nativeElement.width = nativVideoEelement.clientWidth;
    if(this.processedCanvas.nativeElement.height != nativVideoEelement.clientHeight)
      this.processedCanvas.nativeElement.height = nativVideoEelement.clientHeight;
  }
}
