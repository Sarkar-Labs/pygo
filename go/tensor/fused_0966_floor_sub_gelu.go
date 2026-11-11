package tensor

import "fmt"

func Fused0966FloorSubGelu(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_gelu: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.GELU()
	return r, nil
}
func Fused0966FloorSubGeluName() string { return "floor_sub_gelu" }
