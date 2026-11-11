package tensor

import "fmt"

func Fused0961FloorSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Floor()
	r = r.Sign()
	return r, nil
}
func Fused0961FloorSubSignName() string { return "floor_sub_sign" }
