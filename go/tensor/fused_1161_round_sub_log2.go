package tensor

import "fmt"

func Fused1161RoundSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Round()
	r = r.Log2()
	return r, nil
}
func Fused1161RoundSubLog2Name() string { return "round_sub_log2" }
