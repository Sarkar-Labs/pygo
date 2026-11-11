package data

import (
	"encoding/json"
	"fmt"
	"github.com/pygo/go/tensor"
	"os"
)

type COCOAnnotation struct {
	ID, ImageID, CategoryID int
	BBox                    []float64
}
type COCOImage struct {
	ID            int
	FileName      string
	Width, Height int
}
type COCODataset struct {
	Root      string
	Images    map[int]COCOImage
	Anns      map[int][]COCOAnnotation
	IDs       []int
	CatIDs    map[int]int
	Transform Transform
}

func NewCOCO(root, annFile string, transform Transform) (*COCODataset, error) {
	ds := &COCODataset{Root: root, Images: map[int]COCOImage{}, Anns: map[int][]COCOAnnotation{}, CatIDs: map[int]int{}, Transform: transform}
	data, err := os.ReadFile(annFile)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Images      []COCOImage
		Annotations []COCOAnnotation
		Categories  []struct{ ID int }
	}
	if err = json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	for _, img := range raw.Images {
		ds.Images[img.ID] = img
		ds.IDs = append(ds.IDs, img.ID)
	}
	for _, a := range raw.Annotations {
		ds.Anns[a.ImageID] = append(ds.Anns[a.ImageID], a)
	}
	for i, c := range raw.Categories {
		ds.CatIDs[c.ID] = i
	}
	fmt.Printf("COCO: %d images\n", len(ds.Images))
	return ds, nil
}
func (d *COCODataset) Len() int { return len(d.IDs) }
func (d *COCODataset) Get(idx int) *Batch {
	id := d.IDs[idx]
	img := d.Images[id]
	anns := d.Anns[id]
	boxes := make([]float64, 0, len(anns)*4)
	for _, a := range anns {
		boxes = append(boxes, a.BBox...)
	}
	imgT := tensor.Zeros(tensor.Shape{3, int64(img.Height), int64(img.Width)})
	if d.Transform != nil {
		imgT = d.Transform(imgT)
	}
	return &Batch{Inputs: []*tensor.Tensor{imgT}, Targets: []*tensor.Tensor{tensor.FromSlice(boxes, tensor.Shape{int64(len(anns)), 4})}}
}
