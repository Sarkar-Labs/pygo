package tensor

import "fmt"

func Fused0963FloorDivSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("floor_div_sign: nil operand")
	}
	r := x.Div(y)
	r = r.Floor()
	r = r.Sign()
	return r, nil
}
func Fused0963FloorDivSignName() string { return "floor_div_sign" }
