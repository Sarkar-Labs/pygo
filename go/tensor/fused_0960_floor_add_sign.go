package tensor

import "fmt"

func Fused0960FloorAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Floor()
	r = r.Sign()
	return r, nil
}
func Fused0960FloorAddSignName() string { return "floor_add_sign" }
