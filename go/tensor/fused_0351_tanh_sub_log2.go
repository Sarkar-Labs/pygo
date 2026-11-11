package tensor

import "fmt"

func Fused0351TanhSubLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_sub_log2: nil operand")
	}
	r := x.Sub(y)
	r = r.Tanh()
	r = r.Log2()
	return r, nil
}
func Fused0351TanhSubLog2Name() string { return "tanh_sub_log2" }
