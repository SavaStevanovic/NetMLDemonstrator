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
  displayControls = true;
  image: any;
  image1: any;
  processed_context: any;
  lastTime: any;
  readyToSend = true;
  public sockServerResponse: string = ''
  private sockServerResponse$:  Subject<any>
  videoPlaying = false;
  sock: any;

  constructor(
    public frameService: FrameService,
    private filterService: FilterService,
    private sms: SockjsMessageService) {
  }

  capture() {
    var time = this.videoElement.nativeElement.currentTime;
    if (time == 0 || this.videoElement.nativeElement.paused) {
      return
    }

    this.lastTime = time;
    let nativVideoEelement = this.videoElement.nativeElement
    var context = this.unprocessedCanvas.getContext("2d");
    context.drawImage(nativVideoEelement, 0, 0, nativVideoEelement.videoWidth, nativVideoEelement.videoHeight, 0, 0, nativVideoEelement.clientWidth, nativVideoEelement.clientHeight);



    let post_data = {'frame': this.unprocessedCanvas.toDataURL(), 'config': this.filters}
    this.sock.send(JSON.stringify(post_data))
  }

  ngOnInit(): void {
    this.getFilters();
    this.lastTime = 0;
    this.image1 = new Image();
    this.image1.onload = ()=> {
      var processed_context = this.processedCanvas.nativeElement.getContext("2d");
      processed_context.drawImage(this.image1, 0, 0, this.videoElement.nativeElement.clientWidth, this.videoElement.nativeElement.clientHeight, 0, 0, this.videoElement.nativeElement.clientWidth, this.videoElement.nativeElement.clientHeight);
    }
    this.sock = this.frameService.openImageConnection()
    this.sock.onmessage = (v) =>  {
      this.image1.src = JSON.parse(v.data)["image"];
      requestAnimationFrame(this.capture.bind(this))
    }
    this.initCamera({ video: true, audio: true });
  }

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters);
  }

  ngAfterViewInit(): void {
    if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
          this.videoElement.nativeElement.srcObject = stream;
      });
    }


  }

  pause() {
    this.videoElement.nativeElement.pause();
    this.videoPlaying = this.isPlaying();
  }

  resume() {
    this.videoElement.nativeElement.play();
    this.capture()
    this.videoPlaying = this.isPlaying();
  }

  isPlaying() {
    return this?.videoElement?.nativeElement?.paused==false;
  }
  initCamera(config:any) {
    var browser = <any>navigator;

    browser.getUserMedia = (browser.getUserMedia ||
      browser.webkitGetUserMedia ||
      browser.mozGetUserMedia ||
      browser.msGetUserMedia);

    browser.mediaDevices.getUserMedia(config).then(stream => {
      this.videoElement.nativeElement.srcObject = stream;
    });
  }
  updateCanvas(){
    let nativVideoEelement = this.videoElement.nativeElement
    if (this.unprocessedCanvas.width!=nativVideoEelement.clientWidth)
      this.unprocessedCanvas.width = nativVideoEelement.clientWidth;
    if (this.unprocessedCanvas.height!=nativVideoEelement.clientHeight)
      this.unprocessedCanvas.height = nativVideoEelement.clientHeight;
    if (this.processedCanvas.nativeElement.width!=nativVideoEelement.clientWidth)
      this.processedCanvas.nativeElement.width = nativVideoEelement.clientWidth;
    if(this.processedCanvas.nativeElement.height != nativVideoEelement.clientHeight)
      this.processedCanvas.nativeElement.height = nativVideoEelement.clientHeight;
  }
  openSockConn() {
    this.sockServerResponse$ = this.sms.openImageConnection('bla')
    this.sockServerResponse$.subscribe({
      next: (v) => {
        this.sockServerResponse = v.data
        console.log(JSON.stringify(v))
      }
    })
  }
}
