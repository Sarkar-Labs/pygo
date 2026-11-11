package tensor

import "fmt"

func Fused0016AbsSub(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	return r, nil
}
func Fused0016AbsSubName() string { return "abs_sub" }
