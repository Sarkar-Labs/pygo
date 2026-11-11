package tensor

import "fmt"

func Fused1160RoundAddLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_add_log2: nil operand")
	}
	r := x.Add(y)
	r = r.Round()
	r = r.Log2()
	return r, nil
}
func Fused1160RoundAddLog2Name() string { return "round_add_log2" }
