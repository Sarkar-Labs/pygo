package tensor

import "fmt"

func Fused1140RoundAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Sign()
	return r, nil
}
func Fused1140RoundAddSignName() string { return "round_add_sign" }
