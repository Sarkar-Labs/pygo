package tensor

import "fmt"

func Fused0056RoundSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	return r, nil
}
func Fused0056RoundSubName() string { return "round_sub" }
