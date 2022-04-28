import { Component, AfterViewInit, OnInit, ElementRef } from '@angular/core';
import { ViewChild } from '@angular/core';
import { FilterService } from '../../services/filter/filter.service';
import { FrameService } from '../../services/frame/frame.service';
import { Filter } from '../../models/filter';
import { MatSnackBar } from '@angular/material/snack-bar';
import { environment } from '../../../environments/environment';
import { StateService } from '../../services/state/state.service'

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
  processed_context: any;
  unprocessed_context: any;
  video_native_element: any;
  quality: number;
  sock: any;
  production = environment.production;

  constructor(
    public frameService: FrameService,
    private filterService: FilterService,
    private snackBar: MatSnackBar,
    private stateService: StateService) {

  }

  private capture(): void {
    this.setPlaying();
    if (!this.stateService.videoPlaying$) {
      this.toggle_play(false)
      return;
    }

    this.unprocessed_context.drawImage(this.video_native_element, 0, 0, this.video_native_element.clientWidth, this.video_native_element.clientHeight, 0, 0, this.unprocessed_context.canvas.width, this.unprocessed_context.canvas.height);
    if (this.shouldSendReques()) {
      let post_data = { 'frame': this.unprocessedCanvas.toDataURL("image/jpeg", 0.95), 'config': this.filters }
      this.sock.send(JSON.stringify(post_data))
    }
    else {
      this.drawProcessedFrame(null, null, {});
      requestAnimationFrame(this.capture.bind(this));
    }
  }

  private drawProcessedFrame(procesed_frame, mask, data) {
    if (!procesed_frame) {
      procesed_frame = this.video_native_element
    }
    this.processed_context.globalCompositeOperation = 'source-over';
    this.processed_context.drawImage(procesed_frame, 0, 0, this.processedCanvas.nativeElement.width, this.processedCanvas.nativeElement.height);
    if (mask) {
      this.processed_context.globalCompositeOperation = 'multiply';
      this.processed_context.drawImage(mask, 0, 0, this.processedCanvas.nativeElement.width, this.processedCanvas.nativeElement.height);
    }
    if (data['bboxes']) {
      for (let box of data['bboxes']) {
        this.processed_context.beginPath();
        this.processed_context.lineWidth = 3;
        let color = this.toColor(16777216 / (box['category_id'] + 1));
        this.processed_context.strokeStyle = color;
        let bbox = box['bbox'];
        this.processed_context.rect(
          bbox[0] * this.processed_context.canvas.width,
          bbox[1] * this.processed_context.canvas.height,
          bbox[2] * this.processed_context.canvas.width,
          bbox[3] * this.processed_context.canvas.height
        );
        this.processed_context.font = "bold 1.25em Arial";
        this.processed_context.fillStyle = color;
        this.processed_context.fillText(
          box['class'],
          bbox[0] * this.processed_context.canvas.width - 2,
          bbox[1] * this.processed_context.canvas.height - 4);
        this.processed_context.stroke();
      }
    }
    if (data['parts']) {
      this.processed_context.strokeStyle = "black";
      this.processed_context.lineWidth = 3;
      this.processed_context.fillStyle = "blue";
      for (let bpart of data['parts']) {
        this.processed_context.beginPath();
        this.processed_context.arc(
          bpart[0] * this.processed_context.canvas.width,
          bpart[1] * this.processed_context.canvas.height,
          10,
          0,
          Math.PI * 2,
          false);
        this.processed_context.fill()
        this.processed_context.stroke();
      }
    }
    if (data['joints']) {
      this.processed_context.lineWidth = 3;
      this.processed_context.strokeStyle = "blue";
      for (let joint of data['joints']) {
        this.processed_context.beginPath();
        this.processed_context.moveTo(
          joint[0][0] * this.processed_context.canvas.width,
          joint[0][1] * this.processed_context.canvas.height);
        this.processed_context.lineTo(
          joint[1][0] * this.processed_context.canvas.width,
          joint[1][1] * this.processed_context.canvas.height);
        this.processed_context.stroke();
      }
    }
  }

  private imageCollector(expectedCount, completeFn) {
    var receivedCount = 0;
    return function () {
      if (++receivedCount == expectedCount) {
        completeFn();
      }
    };
  };

  ngOnInit(): void {
    this.getFilters();
  }

  private shouldSendReques(): boolean {
    if (this.quality == 0) {
      return false;
    }
    for (let i = 0; i < this.filters.length; i++) {
      if (this.filters[i].selectedModel) {
        return true;
      }
    }
    return false;
  }

  private setupConnection(): void {
    this.sock = this.frameService.openImageConnection(environment.domains.vision.frame_upload_stream);
    this.sock.onmessage = (v) => {
      let data = JSON.parse(v['data'])
      this.processResponse(data);
      requestAnimationFrame(this.capture.bind(this));
    };
  }

  private processResponse(data: any): void {
    let images_to_load = 0
    let procesed_frame = null
    if (data['image']) {
      procesed_frame = new Image()
      procesed_frame.src = data['image']
      data['image'] = null
      images_to_load += 1
    }
    let mask = null
    if (data['mask']) {
      mask = new Image()
      mask.src = data['mask'];
      data['mask'] = null
      images_to_load += 1
    }
    let collector = this.imageCollector(images_to_load, this.drawProcessedFrame.bind(this, procesed_frame, mask, data))
    if (procesed_frame) {
      procesed_frame.onload = collector
    }
    if (mask) {
      mask.onload = collector
    }
  }

  private toColor(num: number): string {
    num >>>= 0;
    var b = num & 0xFF,
      g = (num & 0xFF00) >>> 8,
      r = (num & 0xFF0000) >>> 16;
    return "rgba(" + [r, g, b].join(",") + ")";
  }

  getFilters(): void {
    this.filterService.getFilters()
      .subscribe(filters => this.filters = filters);
  }

  ngAfterViewInit(): void {
    this.processed_context = this.processedCanvas.nativeElement.getContext("2d");
    this.unprocessed_context = this.unprocessedCanvas.getContext("2d");
    this.video_native_element = this.videoElement.nativeElement;
    this.stateService.videoQuality$
      .subscribe(quality => { this.quality = quality; this.updateUnprocessedCanvas() })
    this.videoElement.nativeElement.onplay = () => this.setPlaying()
    this.videoElement.nativeElement.onstop = () => this.setPlaying()
    this.stateService.videoStart$.subscribe(playing => { if (this.filterService.domainSubject.value == "vision") this.toggle_play(playing) })
    this.stateService.menuOpened$.subscribe(opened => this.resizeCanvas())
  }

  toggle_play(play: boolean): void {
    if (play) {
      if (!this.video_native_element.srcObject?.active) {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } }).then(stream => {
          this.video_native_element.srcObject = stream;
          this.setup_stream()
          this.processedCanvas.nativeElement.style.visibility = "visible";
        },
          error => {
            this.snackBar.open('Camera not found.', 'Confirm', {
              duration: 2000
            });
          }
        );
      } else {
        this.setup_stream()
      }
    } else {
      this.video_native_element.pause();
    }

  }

  private setup_stream(): void {
    let is_opened_conn = this.sock?.readyState == WebSocket.OPEN;
    let is_active_camera = this.video_native_element.srcObject?.active;

    if (is_active_camera && is_opened_conn) {
      this.start_stream();
    }

    if (is_active_camera && !is_opened_conn) {
      this.setupConnection();
      this.sock.onopen = this.start_stream.bind(this)
    }
  }

  start_stream(): void {
    this.video_native_element.play();
    this.capture();
  };

  setPlaying(): void {
    this.stateService.setVideoPlaying(this.video_native_element.paused == false
      && this.sock.readyState == WebSocket.OPEN
      && this.video_native_element.srcObject.active)
  }

  updateCanvas(): void {
    this.processedCanvas.nativeElement.width = this.video_native_element.videoWidth;
    this.processedCanvas.nativeElement.height = this.video_native_element.videoHeight;
    this.updateUnprocessedCanvas();
    this.resizeCanvas()
  }

  resizeCanvas(): void {
    var w = this.processedCanvas.nativeElement.parentElement.clientWidth - 20;
    var h = this.processedCanvas.nativeElement.parentElement.clientHeight - 20;
    var aspectRatio = this.video_native_element.videoWidth / this.video_native_element.videoHeight
    if (w / h > aspectRatio) {
      this.processedCanvas.nativeElement.width = h * aspectRatio;
      this.processedCanvas.nativeElement.height = h;
    } else {
      this.processedCanvas.nativeElement.width = w;
      this.processedCanvas.nativeElement.height = w / aspectRatio;
    }
    this.drawProcessedFrame(null, null, {});

  }

  updateUnprocessedCanvas(): void {
    this.unprocessedCanvas.height = this.video_native_element.videoHeight * this.quality;
    this.unprocessedCanvas.width = this.video_native_element.videoWidth * this.quality;
  }
}
