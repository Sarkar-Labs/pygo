package tensor

import "fmt"

func Fused1141RoundSubSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_sign: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Sign()
	return r, nil
}
func Fused1141RoundSubSignName() string { return "round_sub_sign" }
